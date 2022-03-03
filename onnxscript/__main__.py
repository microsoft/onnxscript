# SPDX-License-Identifier: Apache-2.0

import sys
from .converter import convert

# command-line utility to invoke converter on a python file

if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        convert(sys.argv[i])
