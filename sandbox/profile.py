import json
import os
import platform
from pathlib import Path


profile = {
    "python": platform.python_version(),
    "working_directory": os.getcwd(),
}
output_path = Path("/workspace/profile.json")
output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
print(output_path)
