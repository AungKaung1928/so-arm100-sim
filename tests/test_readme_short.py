"""The short README repeats a table and numbers of docs/DETAILS.md, which the other README tests check."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHORT = (ROOT / "README.md").read_text()
DETAILS = (ROOT / "docs" / "DETAILS.md").read_text()


def test_short_rows_are_copies():
    rows = [l for l in SHORT.splitlines() if l.startswith("| `")]
    assert len(rows) == 4
    for l in rows:
        assert l in DETAILS.splitlines(), l


def test_short_numbers_in_details():
    flat = " ".join(DETAILS.split())
    for n in ("median 35.7 (p10–p90 32.6–38.7)", "18.9–25.0"):
        assert n in " ".join(SHORT.split()) and n in flat, n
