# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from typing import Dict, List, Optional, Tuple, Union


def get_parsed_args(
    name: str,
    scenarios: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    epilog: Optional[str] = None,
    new_args: Optional[List[str]] = None,
    **kwargs: Dict[str, Tuple[Union[int, str, float], str]],
) -> argparse.Namespace:
    """
    Returns parsed arguments for examples in this package.

    :param name: script name
    :param scenarios: list of available scenarios
    :param description: parser description
    :param epilog: text at the end of the parser
    :param number: default value for number parameter
    :param repeat: default value for repeat parameter
    :param warmup: default value for warmup parameter
    :param sleep: default value for sleep parameter
    :param expose: if empty, keeps all the parameters,
        if not None, only publish kwargs contains, otherwise the list
        of parameters to publish separated by a comma
    :param new_args: args to consider or None to take `sys.args`
    :param kwargs: additional parameters,
        example: `n_trees=(10, "number of trees to train")`
    :return: parser
    """
    parser = argparse.ArgumentParser(
        prog=name,
        description=description or f"Available options for {name}.py.",
        epilog=epilog or "",
    )
    for k, v in kwargs.items():
        parser.add_argument(
            f"--{k}",
            help=f"{v[1]}, default is {v[0]}",
            type=type(v[0]),
            default=v[0],
        )

    return parser.parse_args(args=new_args)
