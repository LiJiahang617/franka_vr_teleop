import os, sys
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in (_P, os.path.join(_P, 'scripts')):
    if _d not in sys.path:
        sys.path.insert(0, _d)
