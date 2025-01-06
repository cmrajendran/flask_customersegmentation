import sys

def is_virtualenv():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if is_virtualenv():
    print("Running inside a virtual environment.")
else:
    print("Not running inside a virtual environment.")