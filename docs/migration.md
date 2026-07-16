# Migration from `hybrid_architecture_template`

The former flat directory was renamed to `hybrid_qrl` and converted to an
installable `src` layout.

## Import changes

```python
# Before
from hybrid_architecture_template import ConflictGraph, HybridActionHead

# After
from hybrid_qrl import ConflictGraph, HybridActionHead
```

Implementation modules now live under `hybrid_qrl/src/hybrid_qrl`. Tests,
examples, and experiments are intentionally not part of the installed package.

## Command changes

```powershell
# Before
python -m hybrid_architecture_template.example

# After
python .\hybrid_qrl\examples\toy_constrained_action.py
```

```powershell
# Before
python -m hybrid_architecture_template.cartpole_benchmark

# After
python .\hybrid_qrl\experiments\cartpole_benchmark.py
```

Install the renamed package in editable mode before running these commands:

```powershell
python -m pip install -e .\hybrid_qrl
```
