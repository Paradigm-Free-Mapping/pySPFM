# Installation

You'll need to set up a working development environment to use `pySPFM`.
To set up a local environment, you will need Python >=3.8 and the following
packages will need to be installed:

- dask
- dask_jobqueue
- distributed
- nibabel
- nilearn
- numpy
- pylops
- pyproximal
- pywavelets
- scipy

After installing relevant dependencies, you can then install `pySPFM` with:

```bash
pip install pySPFM
```

You can confirm that `pySPFM` has successfully installed by launching a Python instance and running:

```python
import pySPFM
```

You can check that it is available through the command line interface (CLI) with:

```bash
pySPFM --help
```

If no error occurs, `pySPFM` has correctly installed in your environment!
