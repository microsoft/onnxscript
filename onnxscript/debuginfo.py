# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import pprint
import typing


class DebugInfo:

    def __init__(self, lineno, source="string", code=None):
        if hasattr(source, 'source'):
            code = source.source
            current_fn = getattr(source, 'current_fn', None)
            if current_fn is not None:
                source = getattr(source.current_fn, 'name', None)
            else:
                source = None
        if hasattr(lineno, 'lineno'):
            self.ast_obj = lineno
            self.lineno = lineno.lineno
        elif isinstance(lineno, int):
            self.ast_obj = None
            self.lineno = lineno
        elif sys.version_info[:2] < (3, 9):
            # python 3.8 and below
            self.ast_obj = None
            self.lineno = 1
        else:
            raise NotImplementedError(
                f"Unable to extract debug information from type {type(lineno)!r}, "
                f"attributes={pprint.pformat(lineno.__dict__)}.")
        self.source = source
        self.code = None if code is None else code.split('\n')

    def msg(self, text):
        return "ERROR\n%s\n    %s" % (str(self), text)

    def __str__(self):
        if self.code is None:
            line = ''
        else:
            line = "    -- line: " + self.code[self.lineno - 1]
        return "%s:%d%s" % (self.source, self.lineno, line)
