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

## GPU measurements

- Monitor GPU power consumption during all future GPU tests and runs. Preserve timestamped samples and align summaries with the measured workload interval.
- For benchmarks, report average and peak sampled board power, estimated board energy per new token, throughput, and peak allocated VRAM. Record sampling limitations; short runs may not resolve power accurately.
- Include GPU utilization, clocks, temperature, and power/thermal limiting indicators when available. Use read-only monitoring; the desktop safety rules above still apply.
- Treat low power draw as evidence to investigate alongside traces, not as a direct measure of unused compute or a promise of proportional speedup. Optimize verified throughput and efficiency, rather than power draw itself.

## Precision direction

- Native NVFP4 W4A4 training is the user's preferred target when correctness, learning stability, and measured performance support it. Evaluate quantized forward and backward computation; distinguish it from fake-quant QAT, post-training quantization, and frozen quantized bases with adapters.
- Preserve a verified higher-precision reference and a replaceable expert-compute interface while evaluating low-precision backends. Do not assume training quality improves simply because computation uses NVFP4.
