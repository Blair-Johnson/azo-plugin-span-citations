"""Dry-build the plugin in isolated discovery roots using the host inspector."""
import json
from pathlib import Path
import subprocess
import tempfile

import agent_zoo

root = Path(__file__).resolve().parents[1]
host = Path(agent_zoo.__file__).resolve().parents[1]
inspector = host / "skills/agent-zoo-userspace-plugins/scripts/inspect_agent_zoo_plugins.py"

with tempfile.TemporaryDirectory(prefix="span-plugin-inspect-") as tmp:
    common = Path(tmp) / "common"
    common.mkdir()
    (common / "span_citations.py").symlink_to(root / "src/span_citations.py")
    import sys
    result = subprocess.run([sys.executable, str(inspector), "--pipeline", "both",
                             "--plugin-root", tmp, "--init-state", "--json"],
                            text=True, capture_output=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)
    report = json.loads(result.stdout)
    for name, pipeline in report["pipelines"].items():
        serialized = json.dumps(pipeline)
        assert "clip_span" in serialized and "span_citations" in serialized
        print(f"{name}: dry-build passed; clip_span and span_citations discovered")
    print("No live plugin installation was modified.")
