"""Test against the installed Agent Zoo environment, without copying host deps."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
manifest = Path(os.environ.get(
    "AZO_HOST_MANIFEST",
    str(Path.home() / ".local/share/agent-zoo/src/agent-zoo/pixi.toml"),
))
env = dict(os.environ)
env["PYTHONPATH"] = str(root / "src")
command = ["pixi", "run", "--manifest-path", str(manifest), "--frozen",
           "--no-install", "-e", env.get("AZO_HOST_ENVIRONMENT", "agent-zoo-plugins"),
           "python", "-m", "pytest", str(root / "tests"), "-q", *sys.argv[1:]]
raise SystemExit(subprocess.call(command, cwd=root, env=env))
