# Working guidelines

## Git workflow

- `dev` is the primary integration branch.
- For each task or feature, create a separate branch from `dev` before making changes.
- Implement, test, and verify the work on that feature branch. Merge back into `dev` only after the relevant checks pass and the evidence supports correctness.
- If verification fails or material correctness questions remain, keep working on the feature branch; do not merge incomplete work.
- Do not guess or take shortcuts. Distinguish demonstrated results from assumptions and state the limits of verification.
- Preserve unrelated user changes. Stage only files belonging to the task.
- Keep the user's fork (`origin`, `git@github.com:nirbo/mini-AGI.git`) in sync with local work. Push feature branch commits as work progresses, and push `dev` after verified merges.
- Keep unfinished work on its feature branch when pushing. Do not force-push or overwrite remote history without explicit authorization.

## Python environment and desktop safety

- Use only the repository's `.venv` for Python execution and dependency installation.
- If `.venv` is missing, create it with `uv venv .venv`.
- Run Python commands with `.venv/bin/python`, or activate `.venv` first.
- Install dependencies with `uv pip install --python .venv/bin/python ...`.
- Never install Python dependencies into the global or system environment.
- Do not install, upgrade, remove, or reconfigure system packages, drivers, or services as part of project work. This machine is the user's desktop, not a disposable or headless development host.
- If work requires a system change, explain the concrete requirement and obtain explicit authorization before proceeding.

## Quality and communication

- Prioritize accuracy, completeness, code quality, and performance over token savings.
- Investigate thoroughly and use as much reasoning as the task requires. Do not reduce functionality or omit relevant evidence to save effort or tokens.
- Use meaningful tests and verification appropriate to the change. Report what was tested and any important limitations.
- Keep answers and progress updates concise, direct, and easy to read. Lead with the result; avoid repetition and routine tool narration.
