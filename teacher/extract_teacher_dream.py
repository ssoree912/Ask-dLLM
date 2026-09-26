#!/usr/bin/env python
'Extract Dream teacher labels.'

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_teacher import run

if __name__ == "__main__":
    raise SystemExit(run("dream", __doc__))
