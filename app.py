from pathlib import Path
import runpy
import sys


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
runpy.run_path(str(APP_DIR / "web_app.py"), run_name="__main__")
