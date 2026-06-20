import os

from PyInstaller.utils.hooks import collect_submodules


project_root = os.path.abspath(os.path.join(SPECPATH, ".."))
hiddenimports = []
for package in (
    "langgraph",
    "langgraph_checkpoint",
    "langgraph_checkpoint_sqlite",
    "openai",
    "anthropic",
    "boto3",
):
    try:
        hiddenimports.extend(collect_submodules(package))
    except Exception:
        pass

analysis = Analysis(
    [os.path.join(project_root, "desktop_app.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "transformers",
        "ultralytics",
        "easyocr",
        "streamlit",
        "altair",
        "pandas",
        "pyarrow",
        "playwright",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AutoLabel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutoLabel",
)
