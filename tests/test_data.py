"""Tests for data loading and noise functions."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from robust_vision.data.loaders import ScalableDataLoader
from robust_vision.data.noise import NoiseLibrary


class TestScalableDataLoader:
    """Tests for ScalableDataLoader."""
    
    def test_initialization(self):
        """Test data loader initialization."""
        loader = ScalableDataLoader(
            dataset_name="cifar10",
            batch_size=32,
            image_size=(32, 32)
        )
        assert loader.batch_size == 32
        assert loader.image_size == (32, 32)
        assert loader.dataset_name == "cifar10"
    
    def test_load_dataset(self):
        """Test dataset loading."""
        loader = ScalableDataLoader(
            dataset_name="cifar10",
            batch_size=32,
            image_size=(32, 32)
        )
        # Korrekt gebatchtes und geformtes Dataset laden
        ds = loader.create_dataset("train")
        assert ds is not None
        for batch in ds.take(1):
            images, labels = batch
            assert images.shape[0] == 32  # batch size
            assert tuple(images.shape[1:]) == (32, 32, 3)  # image shape
            assert labels.shape[0] == 32


class TestNoiseLibrary:
    """Tests for NoiseLibrary."""
    
    @pytest.fixture
    def noise_lib(self):
        """Create noise library for testing."""
        return NoiseLibrary(rng_key=jax.random.PRNGKey(42))
    
    @pytest.fixture
    def test_images(self):
        """Create test images."""
        return jnp.ones((4, 32, 32, 3)) * 0.5
    
    def test_gaussian_noise(self, noise_lib, test_images):
        """Test Gaussian noise."""
        noisy = noise_lib.gaussian_noise(test_images, std=0.1)
        
        # Check shape preserved
        assert noisy.shape == test_images.shape
        
        # Check values are different (noise was added)
        assert not jnp.allclose(noisy, test_images)
        
        # Check values are in valid range
        assert jnp.all(noisy >= 0.0) and jnp.all(noisy <= 1.0)
    
    def test_salt_pepper_noise(self, noise_lib, test_images):
        """Test salt and pepper noise."""
        noisy = noise_lib.salt_pepper_noise(test_images, prob=0.1)
        
        # Check shape preserved
        assert noisy.shape == test_images.shape
        
        # Check some pixels are 0 or 1
        has_salt = jnp.any(noisy == 1.0)
        has_pepper = jnp.any(noisy == 0.0)
        assert has_salt or has_pepper
    
    def test_fog_noise(self, noise_lib, test_images):
        """Test fog noise."""
        noisy = noise_lib.fog_noise(test_images, intensity=0.3)
        
        # Check shape preserved
        assert noisy.shape == test_images.shape
        
        # Check values are brighter (fog adds white)
        assert jnp.mean(noisy) > jnp.mean(test_images)
        
        # Check values are in valid range
        assert jnp.all(noisy >= 0.0) and jnp.all(noisy <= 1.0)
    
    def test_occlusion_noise(self, noise_lib, test_images):
        """Test occlusion noise."""
        noisy = noise_lib.occlusion_noise(test_images, patch_size=4, num_patches=2)
        
        # Check shape preserved
        assert noisy.shape == test_images.shape
        
        # Check some pixels are 0 (occluded)
        assert jnp.any(noisy == 0.0)
    
    def test_apply_noise(self, noise_lib, test_images):
        """Test apply_noise interface."""
        for noise_type in ["gaussian", "salt_pepper", "fog", "occlusion"]:
            noisy = noise_lib.apply_noise(test_images, noise_type, severity=0.1)
            assert noisy.shape == test_images.shape
            assert jnp.all(noisy >= 0.0) and jnp.all(noisy <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
