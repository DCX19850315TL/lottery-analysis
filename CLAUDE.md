# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

PyCharm-generated Python scaffold. There is no lottery-analysis logic yet: `main.py` only defines `print_hi(name)` and calls it with `'PyCharm'` under the `__main__` guard. No README, tests, linter, or CI configuration exists. Do not assume commands or structure beyond what is listed here.

## Environment

- Python `>=3.14` (venv is CPython 3.14.5).
- `.venv/` is created and managed by `uv`; `pyproject.toml` declares the project (`lottery-analysis`, version `0.1.0`) with no dependencies.
- Windows, Git Bash shell. Venv executables are under `.venv/Scripts/`.

## Commands

```bash
# Install/sync dependencies after editing pyproject.toml
uv sync

# Add a dependency
uv add <package>

# Run the entrypoint
uv run main.py
# or directly
.venv/Scripts/python.exe main.py
```

There is currently no test runner, linter, or formatter configured, so there are no test/lint commands to run. If you add one, record the exact command here.

## Structure

- `main.py` — sole application entrypoint; all executable code is behind `if __name__ == '__main__':`.
- `pyproject.toml` — uv project metadata and dependencies; the only source of truth for the Python version and packages.
- `.idea/` — PyCharm project files; `.idea/workspace.xml` is ignored by `.idea/.gitignore`.
- `.venv/` — local virtual environment; do not edit files inside it.

## Git

The repository has no commits yet on branch `master`. Several `.idea/` and scaffold files are staged but uncommitted.
