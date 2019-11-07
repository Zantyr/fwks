# fwks - the framework for speech processing

This library aims to make training, packaging and deploying sound-related services easy.

### Use cases

View [build/html/modelLifecycle.html][examples in the documentation]. In general, the full lifecycle of the
model should be implemented:

- data preparation
- training
- evaluation - testing and getting metrics
- serialization
- deployment

### Installation

Installation of fwks is two part process. First, core functionality is managed via pip: `python -m pip install --user fwks`.
Second stage is installing extensions via the library itself, vide `python -m fwks status`. Extensions are
installed via `python -m fwks install <ext_name>`


The script may download the required software, but the process is dependent on your environment. Action may be
required to successfully link and compile some of the dependencies.

### Development

Long term goąl list is in [build/html/status.html][status page in docs]. 