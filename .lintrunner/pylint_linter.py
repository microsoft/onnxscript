import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Pattern

IS_WINDOWS: bool = os.name == "nt"


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# adapters/pylint_linter.py:1:0: C0114: Missing module docstring (missing-module-docstring)
RESULTS_RE: Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<code>\S+?):?
    \s(?P<message>.*)
    \s(?:\((?P<string_code>.*)\))
    $
    """
)


def _test_results_re() -> None:
    """
    >>> def t(s): return RESULTS_RE.search(s).groupdict()

    >>> t(r"file.py:40:9: W1514: Using open without explicitly specifying an encoding (unspecified-encoding)")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '40', 'column': '9', 'code': 'W1514',
     'message': 'Using open without explicitly specifying an encoding',
     'string_code': 'unspecified-encoding'}

    >>> t(r"file.py:14:7: R1714: Consider merging these comparisons with 'in' by using 'severity in ('advice', 'disabled')'. Use a set instead if elements are hashable. (consider-using-in)")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '14', 'column': '7', 'code': 'R1714',
     'message': "Consider merging these comparisons with 'in' by using 'severity in ('advice', 'disabled')'. Use a set instead if elements are hashable.",
     'string_code': 'consider-using-in'}

    >>> t(r"file.py:67:15: W1510: Using subprocess.run without explicitly set `check` is not recommended. (subprocess-run-check)")
    ... # doctest: +NORMALIZE_WHITESPACE
    {'file': 'file.py', 'line': '67', 'column': '15', 'code': 'W1510',
     'message': 'Using subprocess.run without explicitly set `check` is not recommended.',
     'string_code': 'subprocess-run-check'}
    """
    pass


def run_command(
    args: List[str],
    *,
    extra_env: Optional[Dict[str, str]],
    retries: int,
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


# Severity can be "I", "C", "R", "W", "E", "F"
# https://pylint.pycqa.org/en/latest/user_guide/usage/output.html
severities = {
    "I": LintSeverity.ADVICE,
    "C": LintSeverity.ADVICE,
    "R": LintSeverity.ADVICE,
    "W": LintSeverity.WARNING,
    "E": LintSeverity.ERROR,
    "F": LintSeverity.ERROR,
}


# https://pylint.pycqa.org/en/latest/user_guide/messages/fatal/method-check-failed.html
def pylint_doc_url(code: str, string_code: str) -> str:
    if code.startswith("I"):
        category = "informational"
    elif code.startswith("C"):
        category = "convention"
    elif code.startswith("R"):
        category = "refactor"
    elif code.startswith("W"):
        category = "warning"
    elif code.startswith("E"):
        category = "error"
    elif code.startswith("F"):
        category = "fatal"
    else:
        return ""

    return (
        f"https://pylint.pycqa.org/en/latest/user_guide/messages/{category}/{string_code}.html"
    )


def check_files(
    filenames: List[str],
    rcfile: Optional[str],
    retries: int,
) -> List[LintMessage]:
    try:
        proc = run_command(
            [sys.executable, "-mpylint", "--score=n"]
            + ([f"--rcfile={rcfile}"] if rcfile else [])
            + filenames,
            extra_env={},
            retries=retries,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="PYLINT",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"]
            + f" ({match['string_code']})"
            + f"\nSee {pylint_doc_url(match['code'], match['string_code'])}",
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code="PYLINT",
            severity=severities.get(match["code"][0], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pylint wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out pylint",
    )
    parser.add_argument(
        "--rcfile",
        default=None,
        type=str,
        help="pylint config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    lint_messages = check_files(list(args.filenames), args.rcfile, args.retries)
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
