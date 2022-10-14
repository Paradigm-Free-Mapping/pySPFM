.. include:: <isonum.txt>

###########################################################
pySPFM
###########################################################

The ``pySPFM`` package is a Python version of AFNI's 3dPFM and 3dMEPFM with new features.

.. image:: https://img.shields.io/pypi/v/pySPFM.svg
   :target: https://pypi.python.org/pypi/pySPFM/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/pySPFM.svg
   :target: https://pypi.python.org/pypi/pySPFM/
   :alt: PyPI - Python Version

.. image:: https://zenodo.org/badge/492450151.svg
   :target: https://zenodo.org/badge/latestdoi/492450151
   :alt: DOI

.. image:: https://circleci.com/gh/eurunuela/pySPFM/tree/main.svg?style=shield
   :target: https://circleci.com/gh/eurunuela/pySPFM/tree/main
   :alt: CircleCI

.. image:: http://img.shields.io/badge/License-LGPL%202.1-blue.svg
   :target: https://opensource.org/licenses/LGPL-2.1
   :alt: License

.. image:: https://readthedocs.org/projects/pyspfm/badge/?version=latest
   :target: http://pyspfm.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/eurunuela/pySPFM/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/eurunuela/pySPFM
   :alt: Codecov

*****
About
*****

``pySPFM`` is a Python implementation of AFNI's 3dPFM and 3dMEPFM with new features.

*********
Citations
*********

When using pySPFM, please include the following citations:

   .. raw:: html

      <script language="javascript">
      var version = 'latest';
      function fillCitation(){
         $('#pySPFM_version').text(version);

         function cb(err, zenodoID) {
            getCitation(zenodoID, 'vancouver-brackets-no-et-al', function(err, citation) {
               $('#pySPFM_citation').text(citation);
            });
            getDOI(zenodoID, function(err, DOI) {
               $('#pySPFM_doi_url').text('https://doi.org/' + DOI);
               $('#pySPFM_doi_url').attr('href', 'https://doi.org/' + DOI);
            });
         }

         if(version == 'latest') {
            getLatestIDFromconceptID("7147858", cb);
         } else {
            getZenodoIDFromTag("7147858", version, cb);
         }
      }
      </script>
      <p>
      <span id="pySPFM_citation">pySPFM</span> 
      This link is for the most recent version of the code and that page has links to DOIs 
      for older versions. To support reproducibility, please cite the version you used: 
      <a id="pySPFM_doi_url" href="https://doi.org/10.5281/zenodo.7147858">https://doi.org/10.5281/zenodo.7147858</a>
      <img src onerror='fillCitation()' alt=""/>
      </p>

      <p>
      2. Caballero Gaudes, C., Petridou, N., Francis, S. T., Dryden, I. L., & Gowland, P. A. (2013).
      <a href="https://doi.org/10.1002/hbm.21452" target="_blank">Paradigm free mapping with sparse regression automatically detects single-trial functional magnetic resonance imaging blood oxygenation level dependent responses.</a>
      <i>Human Brain Mapping</i>.
      </p>

      <p>
      3. Caballero-Gaudes, C., Moia, S., Panwar, P., Bandettini, P. A., & Gonzalez-Castillo, J. (2019).
      <a href="https://doi.org/10.1016/j.neuroimage.2019.116081" target="_blank">A deconvolution algorithm for multi-echo functional MRI: Multi-echo Sparse Paradigm Free Mapping.</a>
      <i>NeuroImage</i>, <i>202</i>, 116081–116081.
      </p>

      <p>
      4. Uruñuela, E., Bolton, T. A. W., Van De Ville, D., & Caballero-Gaudes, C. (2021).
      <a href="http://arxiv.org/abs/2107.12026" target="_blank">Hemodynamic Deconvolution Demystified: Sparsity-Driven Regularization at Work.</a>
      <i>ArXiv</i>, <i>ArXiv:2107.12026</i>.
      </p>

*******************
License Information
*******************

pySPFM is licensed under GNU Lesser General Public License version 2.1.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   hemodynamic deconvolution
   usage
   outputs
   api
   cli


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
