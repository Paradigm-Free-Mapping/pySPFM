import numpy as np
import pytest

from pySPFM.deconvolution.spatial_regularization import clip, generate_delta


def test_clip():
    # Test clipping the input to the atlas
    input_array = np.array([1, 2, 3, 4, 5])
    atlas_array = np.array([1, 1, 1, 2, 2])

    clipped_array = clip(input_array, atlas_array)

    assert np.allclose(
        clipped_array, np.array([0.26726124, 0.53452248, 0.80178373, 0.62469505, 0.78086881])
    )


def test_generate_delta():
    # Test delta generation with dim=2
    h = generate_delta(dim=2)
    assert h.shape == (3, 3)

    # Test delta generation with dim=3
    h = generate_delta(dim=3)
    assert h.shape == (3, 3, 3)

    # Test delta generation with dim different from 2 or 3
    pytest.raises(ValueError, generate_delta, dim=1)
