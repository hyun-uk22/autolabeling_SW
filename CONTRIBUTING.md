# Contributing

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

Specialist model dependencies are optional:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-specialists.txt
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q desktop_app.py web_app.py src tests
```

Run Streamlit locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_app.py
```

## Pull Requests

- Keep changes scoped and preserve existing CLI, desktop, and Streamlit behavior where applicable.
- Add or update tests for behavioral changes.
- Update README or the relevant specification document when a command, option, format, or workflow changes.
- Never commit credentials, datasets, generated labels, model weights, build output, or local workspace configuration.
- Confirm `git status --short --ignored` and `git diff --check` before opening a pull request.

## Windows Installer

```powershell
.\packaging\build.ps1
```

The generated `build/`, `dist/`, and `dist-installer/` directories are intentionally ignored. Publish signed installers through GitHub Releases rather than committing binaries to the repository.
