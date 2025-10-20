import json
import os
from urllib.request import urlopen, urlretrieve

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipintegration", action="store_true", default=False, help="Skip integration tests."
    )


@pytest.fixture
def skip_integration(request):
    return request.config.getoption("--skipintegration")


def fetch_file(osf_id, path, filename):
    """
    Fetches file located on OSF and downloads to `path`/`filename`1
    Parameters
    ----------
    osf_id : str
        Unique OSF ID for file to be downloaded. Will be inserted into relevant
        location in URL: https://osf.io/{osf_id}/download
    path : str
        Path to which `filename` should be downloaded. Ideally a temporary
        directory
    filename : str
        Name of file to be downloaded (does not necessarily have to match name
        of file on OSF)
    Returns
    -------
    full_path : str
        Full path to downloaded `filename`
    """
    # Use OSF API v2 to get the download URL
    api_url = f"https://api.osf.io/v2/files/{osf_id}/"
    full_path = os.path.join(path, filename)

    if not os.path.isfile(full_path):
        # Fetch metadata to get download link
        with urlopen(api_url) as response:
            metadata = json.load(response)
            download_url = metadata["data"]["links"]["download"]

        # Download the actual file
        urlretrieve(download_url, full_path)

    return full_path


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """Test path that will be used to download all files"""
    return tmp_path_factory.getbasetemp()


@pytest.fixture
def pylops_results(testpath):
    return fetch_file("bmhtr", testpath, "pylops_fista.npy")


@pytest.fixture
def sim_data(testpath):
    return fetch_file("789z6", testpath, "sim_data.npy")


@pytest.fixture
def sim_hrf(testpath):
    return fetch_file("j2u6n", testpath, "sim_hrf.npy")


@pytest.fixture
def mask_five_echo(testpath):
    return fetch_file("jv5fn", testpath, "mask_five_echo.nii.gz")


@pytest.fixture
def spm_single_echo_block(testpath):
    return fetch_file("4k85j", testpath, "spm_single_echo_block.npy")


@pytest.fixture
def spm_single_echo(testpath):
    return fetch_file("zt3pm", testpath, "spm_single_echo.npy")


@pytest.fixture
def glover_multi_echo(testpath):
    return fetch_file("xke79", testpath, "glover_multi_echo.npy")


@pytest.fixture
def nih_five_echo_1(testpath):
    return fetch_file("em9r8", testpath, "p06.SBJ01_S09_Task11_e1.sm.nii.gz")


@pytest.fixture
def fista_results(testpath):
    return fetch_file("3a925", testpath, "fista_results.npy")


@pytest.fixture
def fista_positives(testpath):
    return fetch_file("9jbas", testpath, "fista_positives.npy")


@pytest.fixture
def coef_path_results(testpath):
    return fetch_file("nxgeq", testpath, "coef_path.npy")


@pytest.fixture
def test_AUC(testpath):
    return fetch_file("spzu9", testpath, "test_AUC.nii.gz")


@pytest.fixture
def mean_AUC(testpath):
    return fetch_file("vhdm9", testpath, "mean_AUC.nii.gz")


@pytest.fixture
def auc_4D_thr(testpath):
    return fetch_file("2pqmy", testpath, "test_AUC_4D_thr.nii.gz")


@pytest.fixture
def five_echo_e1(testpath):
    return fetch_file("u7bej", testpath, "p06.SBJ01_S09_Task11_e1.psc.nii.gz")


@pytest.fixture
def five_echo_e2(testpath):
    return fetch_file("a652w", testpath, "p06.SBJ01_S09_Task11_e2.psc.nii.gz")


@pytest.fixture
def five_echo_e3(testpath):
    return fetch_file("tjq4c", testpath, "p06.SBJ01_S09_Task11_e3.psc.nii.gz")


@pytest.fixture
def five_echo_e4(testpath):
    return fetch_file("ukmsp", testpath, "p06.SBJ01_S09_Task11_e4.psc.nii.gz")


@pytest.fixture
def five_echo_e5(testpath):
    return fetch_file("cqk2m", testpath, "p06.SBJ01_S09_Task11_e5.psc.nii.gz")
