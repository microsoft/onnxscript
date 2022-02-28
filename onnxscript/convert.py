import sys
from converter import convert

if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        convert(sys.argv[i])