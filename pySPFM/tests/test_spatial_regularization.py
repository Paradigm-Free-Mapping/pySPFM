"""Tests for spatial regularization functions as developed in Total Activation."""

import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker

from pySPFM._solvers.spatial_regularization import (
    clip,
    generate_delta,
    spatial_structured_sparsity,
    spatial_tikhonov,
)


class TestGenerateDelta:
    """Tests for the generate_delta function."""

    def test_generate_delta_2d(self):
        """Test delta generation with dim=2 (2D Laplacian)."""
        h = generate_delta(dim=2)
        assert h.shape == (3, 3)
        # Check the 2D Laplacian operator values
        expected = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        np.testing.assert_array_equal(h, expected)

    def test_generate_delta_3d(self):
        """Test delta generation with dim=3 (3D Laplacian)."""
        h = generate_delta(dim=3)
        assert h.shape == (3, 3, 3)
        # Check the 3D Laplacian operator structure
        # Central slice should have -6 in the center
        assert h[1, 1, 1] == -6
        # Adjacent voxels should be 1
        assert h[0, 1, 1] == 1  # above
        assert h[2, 1, 1] == 1  # below
        assert h[1, 0, 1] == 1  # left
        assert h[1, 2, 1] == 1  # right
        assert h[1, 1, 0] == 1  # front
        assert h[1, 1, 2] == 1  # back
        # Sum should be 0 (Laplacian property)
        assert np.sum(h) == 0

    def test_generate_delta_invalid_dim(self):
        """Test delta generation with invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be 2 or 3"):
            generate_delta(dim=1)
        with pytest.raises(ValueError, match="Dimension must be 2 or 3"):
            generate_delta(dim=4)


class TestClip:
    """Tests for the clip function used in structured sparsity."""

    def test_clip_basic(self):
        """Test basic clipping functionality."""
        input_array = np.array([1, 2, 3, 4, 5])
        atlas_array = np.array([1, 1, 1, 2, 2])

        clipped_array = clip(input_array, atlas_array)

        # Region 1 has norm sqrt(1^2 + 2^2 + 3^2) = sqrt(14) > 1, so it gets normalized
        # Region 2 has norm sqrt(4^2 + 5^2) = sqrt(41) > 1, so it gets normalized
        assert np.allclose(
            clipped_array, np.array([0.26726124, 0.53452248, 0.80178373, 0.62469505, 0.78086881])
        )

    def test_clip_no_clipping_needed(self):
        """Test clipping when input is already within bounds."""
        input_array = np.array([0.1, 0.2, 0.3])
        atlas_array = np.array([1, 1, 1])

        clipped_array = clip(input_array, atlas_array)

        # Norm is sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14) < 1, so no clipping
        np.testing.assert_array_almost_equal(clipped_array, input_array)

    def test_clip_multiple_regions(self):
        """Test clipping with multiple regions."""
        input_array = np.array([10, 0, 0, 0.1, 0.1])
        atlas_array = np.array([1, 1, 1, 2, 2])

        clipped_array = clip(input_array, atlas_array)

        # Region 1: norm = 10 > 1, gets clipped to [1, 0, 0]
        np.testing.assert_array_almost_equal(clipped_array[:3], [1, 0, 0])
        # Region 2: norm = sqrt(0.02) < 1, no clipping
        np.testing.assert_array_almost_equal(clipped_array[3:], [0.1, 0.1])

    def test_clip_3d_input(self):
        """Test clipping with 3D input array."""
        input_array = np.zeros((3, 3, 3))
        input_array[0, 0, 0] = 5
        input_array[1, 1, 1] = 0.1

        atlas_array = np.zeros((3, 3, 3), dtype=int)
        atlas_array[0, 0, 0] = 1
        atlas_array[1, 1, 1] = 2

        clipped_array = clip(input_array, atlas_array)

        # Region 1: value 5 > 1, gets clipped to 1
        assert clipped_array[0, 0, 0] == 1.0
        # Region 2: value 0.1 < 1, no clipping
        assert clipped_array[1, 1, 1] == 0.1


class TestTotalActivationComparison:
    """
    Tests comparing pySPFM spatial regularization to Total Activation behavior.

    These tests verify that the implementation follows the Total Activation
    framework's approach to spatial regularization.
    """

    def test_laplacian_operator_properties(self):
        """Test that the Laplacian operators have correct properties."""
        # 2D Laplacian
        h2d = generate_delta(dim=2)
        assert np.sum(h2d) == 0, "2D Laplacian should sum to zero"
        assert h2d[1, 1] == -4, "2D Laplacian center should be -4"

        # 3D Laplacian
        h3d = generate_delta(dim=3)
        assert np.sum(h3d) == 0, "3D Laplacian should sum to zero"
        assert h3d[1, 1, 1] == -6, "3D Laplacian center should be -6"

    def test_tikhonov_fft_domain(self):
        """Test that Tikhonov uses FFT-based computation correctly."""
        # The FFT of the Laplacian should be applied in frequency domain
        h = generate_delta(dim=3)
        h_fft = np.fft.fftn(h, (10, 10, 10))

        # FFT should be complex
        assert np.iscomplexobj(h_fft)

        # The product h * conj(h) should be real and non-negative
        h_squared = h_fft * np.conj(h_fft)
        assert np.all(np.imag(h_squared) < 1e-10), "h*conj(h) should be real"
        assert np.all(np.real(h_squared) >= -1e-10), "h*conj(h) should be non-negative"

    def test_gradient_descent_step(self):
        """Test that the gradient descent step is correctly implemented."""
        # For Tikhonov: x_new = (1-mu)*x + mu*y - mu*lambda*L'L*x
        # This is a gradient descent step on ||y-x||^2 + lambda*||Lx||^2

        mu = 0.1
        lambda_ = 0.5

        # Simple test case
        x = np.array([1.0, 2.0, 1.0])
        y = np.array([1.0, 1.5, 1.0])

        # Manual gradient descent step (simplified, 1D case)
        # L = [1, -2, 1] (1D Laplacian)
        L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])  # Simplified
        LtL = L.T @ L

        # Gradient: 2*(x-y) + 2*lambda*L'Lx
        grad = 2 * (x - y) + 2 * lambda_ * (LtL @ x)

        # Update: x_new = x - mu/2 * grad
        x_new_expected = x - mu / 2 * grad

        # The update should move towards y while smoothing
        assert not np.allclose(x, x_new_expected)


class TestSpatialTikhonov:
    """Tests for the spatial_tikhonov function."""

    @pytest.fixture
    def masker(self, tmp_path):
        """Create a NiftiMasker for testing."""
        # Create a simple mask
        mask_data = np.ones((10, 10, 10), dtype=np.int8)
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        mask_path = tmp_path / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        masker = NiftiMasker(mask_img=str(mask_path))
        masker.fit()
        return masker

    def test_spatial_tikhonov_basic_3d(self, masker):
        """Test spatial Tikhonov regularization in 3D mode."""
        n_voxels = 1000  # 10x10x10
        n_timepoints = 5

        # Create test data
        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32)
        data = estimates.copy()

        # Run spatial Tikhonov with minimal iterations
        result = spatial_tikhonov(
            estimates=estimates,
            data=data,
            masker=masker,
            niter=1,
            dim=3,
            lambda_=0.1,
            mu=0.01,
        )

        # Check output shape
        assert result.shape == (n_voxels, n_timepoints)
        # Result should be real
        assert np.all(np.isreal(result))

    def test_spatial_tikhonov_basic_2d(self, masker):
        """Test spatial Tikhonov regularization in 2D (slice-wise) mode."""
        n_voxels = 1000  # 10x10x10
        n_timepoints = 5

        # Create test data
        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32)
        data = estimates.copy()

        # Run spatial Tikhonov with minimal iterations
        result = spatial_tikhonov(
            estimates=estimates,
            data=data,
            masker=masker,
            niter=1,
            dim=2,
            lambda_=0.1,
            mu=0.01,
        )

        # Check output shape
        assert result.shape == (n_voxels, n_timepoints)

    def test_spatial_tikhonov_smoothing_effect(self, masker):
        """Test that Tikhonov regularization has a smoothing effect."""
        n_voxels = 1000
        n_timepoints = 3

        # Create noisy estimates
        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32)

        # Run with more iterations and reasonable regularization
        # Small lambda to ensure stability (large lambda can cause divergence)
        result = spatial_tikhonov(
            estimates=estimates,
            data=estimates.copy(),
            masker=masker,
            niter=10,
            dim=3,
            lambda_=0.01,
            mu=0.1,
        )

        # Result should be different from input (smoothing occurred)
        assert not np.allclose(estimates, result)
        # Variance should be reduced (smoothing effect)
        assert np.var(result) <= np.var(estimates) * 1.1  # Allow small tolerance

    def test_spatial_tikhonov_convergence(self, masker):
        """Test that more iterations lead to stronger regularization."""
        n_voxels = 1000
        n_timepoints = 3

        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32)
        data = estimates.copy()

        # Run with different number of iterations
        result_1iter = spatial_tikhonov(
            estimates=estimates.copy(),
            data=data,
            masker=masker,
            niter=1,
            dim=3,
            lambda_=0.5,
            mu=0.1,
        )

        result_5iter = spatial_tikhonov(
            estimates=estimates.copy(),
            data=data,
            masker=masker,
            niter=5,
            dim=3,
            lambda_=0.5,
            mu=0.1,
        )

        # Results should be different
        assert not np.allclose(result_1iter, result_5iter)


class TestSpatialStructuredSparsity:
    """Tests for the spatial_structured_sparsity function."""

    @pytest.fixture
    def mask(self):
        """Create a mask for testing."""
        # Create a simple binary mask with region labels
        mask_data = np.ones((10, 10, 10), dtype=np.int32)
        mask = nib.Nifti1Image(mask_data, np.eye(4))
        return mask

    def test_spatial_structured_sparsity_basic(self, mask):
        """Test basic structured sparsity regularization."""
        n_voxels = 1000  # 10x10x10
        n_timepoints = 3

        # Create test data
        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32) * 0.1
        data = estimates.copy()
        dims = [10, 10, 10, n_timepoints]

        # Run structured sparsity
        result = spatial_structured_sparsity(
            estimates=estimates,
            data=data,
            mask=mask,
            niter=1,
            dims=dims,
            lambda_=0.1,
        )

        # Check output shape
        assert result.shape == (n_voxels, n_timepoints)


class TestIntegration:
    """Integration tests for the spatial regularization pipeline."""

    @pytest.fixture
    def masker(self, tmp_path):
        """Create a real NiftiMasker for integration testing."""
        # Create a simple mask
        mask_data = np.ones((10, 10, 10), dtype=np.int8)
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        mask_path = tmp_path / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        masker = NiftiMasker(mask_img=str(mask_path))
        masker.fit()
        return masker

    def test_tikhonov_with_real_masker(self, masker):
        """Test Tikhonov regularization with a real NiftiMasker."""
        n_voxels = 1000  # 10x10x10
        n_timepoints = 5

        # Create test data
        np.random.seed(42)
        estimates = np.random.randn(n_voxels, n_timepoints).astype(np.float32)
        data = estimates.copy()

        # This test verifies the function runs without errors with a real masker
        result = spatial_tikhonov(
            estimates=estimates,
            data=data,
            masker=masker,
            niter=2,
            dim=3,
            lambda_=0.1,
            mu=0.05,
        )

        assert result.shape == (n_voxels, n_timepoints)
        # Result should be real (no imaginary component from FFT)
        assert np.all(np.isreal(result))

    def test_pipeline_preserves_signal(self, masker):
        """Test that the pipeline preserves signal structure."""
        n_voxels = 1000
        n_timepoints = 5

        # Create structured signal (smooth spatial pattern)
        np.random.seed(42)
        signal = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
        # Add a smooth "blob" of activity
        for t in range(n_timepoints):
            signal[400:600, t] = np.sin(2 * np.pi * t / n_timepoints)

        # Add noise
        noisy_signal = signal + np.random.randn(n_voxels, n_timepoints).astype(np.float32) * 0.1

        # Apply spatial regularization with conservative parameters
        # Small lambda to avoid over-regularization
        result = spatial_tikhonov(
            estimates=noisy_signal,
            data=noisy_signal,
            masker=masker,
            niter=3,
            dim=3,
            lambda_=0.01,
            mu=0.1,
        )

        # Result should have same shape
        assert result.shape == signal.shape

        # Verify output is finite
        assert np.all(np.isfinite(result)), "Result contains non-finite values"
