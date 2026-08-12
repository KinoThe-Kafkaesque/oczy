"""Module entrypoint for ``python -m oczy.experiments.meta_cortex``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
