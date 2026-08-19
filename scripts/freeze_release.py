"""G3/G4: SHA256 freeze of v1 splits and locked log markdown.

Does not touch model weights. Writes:
  data/splits/v1/SHA256SUMS.txt
  data/splits/v2/SHA256SUMS.txt  (if present)
  logs/LOCK.md
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_LOGS = {"demo.md", "LOCK.md", "goated_checklist.md"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(folder: Path, patterns: tuple[str, ...]) -> list[tuple[str, str]]:
    seen: set[Path] = set()
    rows = []
    for pat in patterns:
        for p in sorted(folder.glob(pat)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(ROOT).as_posix()
            rows.append((sha256_file(p), rel))
    rows.sort(key=lambda r: r[1])
    return rows


def write_sumfile(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {rel}" for digest, rel in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    v1 = ROOT / "data" / "splits" / "v1"
    if not v1.exists():
        raise SystemExit(f"missing {v1}")
    v1_rows = hash_tree(v1, ("*.jsonl", "*.json", "index.json"))
    # exclude our own sums file if re-run
    v1_rows = [r for r in v1_rows if not r[1].endswith("SHA256SUMS.txt")]
    write_sumfile(v1 / "SHA256SUMS.txt", v1_rows)

    v2 = ROOT / "data" / "splits" / "v2"
    v2_rows: list[tuple[str, str]] = []
    if v2.exists():
        v2_rows = hash_tree(v2, ("*.jsonl", "*.json"))
        v2_rows = [r for r in v2_rows if not r[1].endswith("SHA256SUMS.txt")]
        write_sumfile(v2 / "SHA256SUMS.txt", v2_rows)

    log_dir = ROOT / "logs"
    log_rows = []
    for p in sorted(log_dir.glob("*.md")):
        if p.name in SKIP_LOGS:
            continue
        log_rows.append((sha256_file(p), p.relative_to(ROOT).as_posix()))

    today = date.today().isoformat()
    lock = [
        f"# Log + split freeze ({today})",
        "",
        "Appendix archive of locked markdown receipts. **Do not regenerate `data/splits/v1/`.**",
        "Paper numbers: `logs/findings.md`. Draft: `paper/main.tex`.",
        "",
        "Weights (`*.safetensors`) are **not** in git (`runs/` is gitignored). Re-FT seed 1:",
        "",
        "```powershell",
        "python scripts\\finetune_ar.py --seed 1 --out-dir runs\\ar_ft",
        "```",
        "",
        "Suggested git tag after committing this freeze: `workshop-draft-2026-08-19`.",
        "",
        "## data/splits/v1",
        "",
        "| sha256 | file |",
        "|---|---|",
    ]
    for digest, rel in v1_rows:
        lock.append(f"| `{digest}` | `{rel}` |")
    if v2_rows:
        lock += ["", "## data/splits/v2 (multi-query; v1 untouched)", "", "| sha256 | file |", "|---|---|"]
        for digest, rel in v2_rows:
            lock.append(f"| `{digest}` | `{rel}` |")
    lock += ["", "## logs/*.md (locked receipts; demo.md excluded)", "", "| sha256 | file |", "|---|---|"]
    for digest, rel in log_rows:
        lock.append(f"| `{digest}` | `{rel}` |")
    lock.append("")
    (log_dir / "LOCK.md").write_text("\n".join(lock), encoding="utf-8")
    print(f"v1 files {len(v1_rows)}  v2 files {len(v2_rows)}  logs {len(log_rows)}")
    print("wrote data/splits/v1/SHA256SUMS.txt")
    if v2_rows:
        print("wrote data/splits/v2/SHA256SUMS.txt")
    print("wrote logs/LOCK.md")


if __name__ == "__main__":
    sys.exit(main() or 0)
