from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app


def main() -> None:
    schema = app.openapi()
    output_path = ROOT / "backend" / "openapi.json"
    output_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote OpenAPI schema to {output_path}")


if __name__ == "__main__":
    main()
