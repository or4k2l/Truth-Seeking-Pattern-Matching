"""Tests for model architecture."""

import pytest
import jax
import jax.numpy as jnp

from robust_vision.models.cnn import ProductionCNN, ResidualBlock


class TestProductionCNN:
    """Tests for ProductionCNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return ProductionCNN(
            n_classes=10,
            features=[64, 128],
            dropout_rate=0.3,
            use_residual=True
        )
    
    @pytest.fixture
    def test_input(self):
        """Create test input."""
        return jnp.ones((4, 32, 32, 3))
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model.n_classes == 10
        assert model.features == [64, 128]
        assert model.dropout_rate == 0.3
    
    def test_forward_pass(self, model, test_input):
        """Test forward pass produces correct output shape."""
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, test_input, training=False)
        
        # Forward pass
        logits = model.apply(params, test_input, training=False)
        
        # Check output shape
        assert logits.shape == (4, 10)  # (batch_size, n_classes)
    
    def test_training_mode(self, model, test_input):
        """Test model works in training mode."""
        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, test_input, training=False)
        
        # Forward pass in training mode with mutable batch_stats
        output = model.apply(variables, test_input, training=True, 
                           mutable=['batch_stats'], rngs={'dropout': rng})
        
        # Extract logits and updated batch_stats
        logits, updated_vars = output
        
        # Verify batch_stats is present (model uses BatchNorm)
        assert 'batch_stats' in updated_vars
        
        # Should produce valid output
        assert logits.shape == (4, 10)
        assert not jnp.any(jnp.isnan(logits))
    
    def test_different_input_sizes(self, model):
        """Test model works with different input sizes."""
        rng = jax.random.PRNGKey(0)
        
        for size in [28, 32, 64]:
            test_input = jnp.ones((2, size, size, 3))
            params = model.init(rng, test_input, training=False)
            logits = model.apply(params, test_input, training=False)
            
            assert logits.shape == (2, 10)
    
    def test_no_residual(self, test_input):
        """Test model without residual connections."""
        model = ProductionCNN(
            n_classes=10,
            features=[64, 128],
            dropout_rate=0.0,
            use_residual=False
        )
        
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, test_input, training=False)
        logits = model.apply(params, test_input, training=False)
        
        assert logits.shape == (4, 10)
    
    def test_jit_compilation(self, model, test_input):
        """Test model can be JIT compiled."""
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, test_input, training=False)
        
        # JIT compile
        @jax.jit
        def predict(params, x):
            return model.apply(params, x, training=False)
        
        # Run
        logits = predict(params, test_input)
        assert logits.shape == (4, 10)


class TestResidualBlock:
    """Tests for ResidualBlock."""
    
    def test_residual_block(self):
        """Test ResidualBlock forward pass."""
        block = ResidualBlock(features=64, dropout_rate=0.1)
        
        rng = jax.random.PRNGKey(0)
        test_input = jnp.ones((2, 32, 32, 64))
        
        params = block.init(rng, test_input, training=False)
        output = block.apply(params, test_input, training=False)
        
        # Output shape should match input shape (same features)
        assert output.shape == test_input.shape


if __name__ == "__main__":
    pytest.main([__file__])
