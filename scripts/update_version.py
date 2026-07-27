from pathlib import Path
from datetime import datetime
import subprocess

root = Path(__file__).resolve().parents[1]

date = subprocess.check_output(
    ["git", "log", "-1", "--format=%cI"],
    cwd=root,
    text=True,
).strip()

dt = datetime.fromisoformat(date.replace("Z", "+00:00"))

version = f"{dt.year}.{dt.month}.{dt.day}"

(root / "lace" / "_version.py").write_text(f'__version__ = "{version}"\n')

print(version)
