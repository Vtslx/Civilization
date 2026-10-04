"""Allow ``python -m civilization ...`` in addition to the ``civilization`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
