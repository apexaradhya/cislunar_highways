"""Inline web/explorer_data.json into the explorer template -> docs/explorer/cislunar_explorer.html."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
tpl = (root / "web" / "explorer_template.html").read_text()
data = (root / "web" / "explorer_data.json").read_text().replace("</", "<\\/")
out = root / "docs" / "explorer" / "cislunar_explorer.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tpl.replace("__DATA__", data))
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
