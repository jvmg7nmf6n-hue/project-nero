from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nero_app.core.research_integrity import write_research_integrity_reports


def main() -> None:
    written = write_research_integrity_reports()
    print("Research integrity reports written:")
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
