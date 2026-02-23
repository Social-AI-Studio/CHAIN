"""
chainbench: CHAIN – Causal Hierarchy of Actions and Interactions

An interactive 3D physics-driven benchmark for evaluating whether Vision-Language
Models can understand, plan, and execute structured action sequences grounded in
physical constraints.

Supported tasks:
- Luban Lock disassembly (Unity-backed interlocking puzzle environment)
- Stacking Game (polycube packing under geometric constraints)

Example Usage:
```python
from chainbench.core.config import load_config
from chainbench.runner import BenchmarkRunner

config = load_config("eval_configs/luban.yaml")
runner = BenchmarkRunner(config)
runner.setup()
runner.run_benchmark(num_runs=5)
```

Command-line Usage:
```bash
chainbench run --config eval_configs/luban.yaml
chainbench benchmark --config eval_configs/luban.yaml --num-runs 5
chainbench evaluate --results-dir logs/
```
"""

# Normal imports instead of lazy loading to ensure proper registry initialization
from chainbench.core.config import Config, load_config, validate_config
from chainbench.runner import BenchmarkRunner

__version__ = "0.1.0"
__author__ = "Maojia Song"
__email__ = "maojia_song@mymail.sutd.edu.sg"

__all__ = [
    "Config",
    "load_config",
    "validate_config",
    "BenchmarkRunner"
]
