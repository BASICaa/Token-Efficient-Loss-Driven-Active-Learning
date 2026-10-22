# Repository execution environment

Use the Conda environment `hf-agents` for Python, tests, formatting and dependency
installation. Use `conda run -n hf-agents ...` when it is not activated. Do not
install dependencies into base Conda or system Python.

The beta application and controlled experiment engine have independent Python
packages named `TEFLD`. Run their tests in separate processes and directories:

```powershell
conda run -n hf-agents python -m pytest tests
cd experiments/controlled
conda run -n hf-agents python -m pytest tests
```

Keep the measured results under `results/` separate from future experiment
outputs. Offline checks must not start real training or paid API calls. Changes
to experiment code require fresh run identities; do not rewrite old results.
