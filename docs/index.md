```{eval-rst}
.. include:: <isonum.txt>
```

# pySPFM

The `pySPFM` package is a Python version of AFNI's 3dPFM and 3dMEPFM with new features.

```{image} https://img.shields.io/pypi/v/pySPFM.svg
:alt: Latest Version
:target: https://pypi.python.org/pypi/pySPFM/
```

```{image} https://img.shields.io/pypi/pyversions/pySPFM.svg
:alt: PyPI - Python Version
:target: https://pypi.python.org/pypi/pySPFM/
```

```{image} https://zenodo.org/badge/492450151.svg
:alt: DOI
:target: https://zenodo.org/badge/latestdoi/492450151
```

```{image} https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main.svg?style=shield
:alt: CircleCI
:target: https://circleci.com/gh/Paradigm-Free-Mapping/pySPFM/tree/main
```

```{image} http://img.shields.io/badge/License-LGPL%202.1-blue.svg
:alt: License
:target: https://opensource.org/licenses/LGPL-2.1
```

```{image} https://readthedocs.org/projects/pyspfm/badge/?version=latest
:alt: Documentation Status
:target: http://pyspfm.readthedocs.io/en/latest/?badge=latest
```

```{image} https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM/branch/main/graph/badge.svg
:alt: Codecov
:target: https://codecov.io/gh/Paradigm-Free-Mapping/pySPFM
```

## About

`pySPFM` is a Python implementation of AFNI's 3dPFM and 3dMEPFM with new features.

## Citations

When using pySPFM, please include the following citations:

> ```{raw} html
> <script language="javascript">
> var version = 'latest';
> function fillCitation(){
>    $('#pySPFM_version').text(version);
>
>    function cb(err, zenodoID) {
>       getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
>          $('#pySPFM_citation').text(citation);
>       });
>       getDOI(zenodoID, function(err, DOI) {
>          $('#pySPFM_doi_url').text('https://doi.org/' + DOI);
>          $('#pySPFM_doi_url').attr('href', 'https://doi.org/' + DOI);
>       });
>    }
>
>    if(version == 'latest') {
>       getLatestIDFromconceptID("6600095", cb);
>    } else {
>       getZenodoIDFromTag("6600095", version, cb);
>    }
> }
> </script>
> <p>
> <span id="pySPFM_citation">pySPFM</span>
> This link is for the most recent version of the code and that page has links to DOIs
> for older versions. To support reproducibility, please cite the version you used:
> <a id="pySPFM_doi_url" href="https://doi.org/10.5281/zenodo.6600095">https://doi.org/10.5281/zenodo.6600095</a>
> <img src onerror='fillCitation()' alt=""/>
> </p>
>
> <p>
> 2. Caballero Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013).
> <a href="https://doi.org/10.1002/hbm.21452" target="_blank">Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses.</a>
> <i>Human Brain Mapping</i>.
> </p>
>
> <p>
> 3. Caballero-Gaudes, C., Moia, S., Panwar, P., Bandettini, P. A., & Gonzalez-Castillo, J. (2019).
> <a href="https://doi.org/10.1016/j.neuroimage.2019.116081" target="_blank">A deconvolution algorithm for multi-echo functional MRI: Multi-echo Sparse Paradigm Free Mapping.</a>
> <i>NeuroImage</i>, <i>202</i>, 116081–116081.o
> </p>
>
> <p>
> 4. Uruñuela, E., Bolton, T. A. W., Van De Ville, D., & Caballero-Gaudes, C. (2023).
> <a href="https://doi.org/10.52294/001c.87574" target="_blank">Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work.</a>
> <i>Aperture Neuro</i>, <i>vol. 3, Aug. 2023, pp. 1–25</i>.
> </p>
> ```

## License Information

pySPFM is licensed under GNU Lesser General Public License version 2.1.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

installation
hemodynamic deconvolution
usage
outputs
api
examples
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
