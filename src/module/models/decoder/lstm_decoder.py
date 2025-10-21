"""
Simple decoder heads for LSTM baseline

Provides lightweight MLP-based decoders for LSTM output
Compatible with both single-target and series prediction tasks

Author: For LSTM baseline comparison
Date: 2025-10-14
"""

import torch
import torch.nn as nn


class LSTMRegressionHead(nn.Module):
    """
    Simple MLP regression head for LSTM encoder output

    Takes LSTM hidden state and predicts continuous values

    Args:
        input_dim: LSTM output dimension (e.g., 256)
        num_targets: Number of regression targets (e.g., 7 emotions)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_targets: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets)
        )

        print(f"LSTM Regression Head initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {num_targets}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, input_dim) - LSTM encoder output

        Returns:
            (batch, num_targets) - predicted values
        """
        return self.mlp(x)  # (batch, num_targets)


class LSTMSeriesRegressionHead(nn.Module):
    """
    Series regression head for LSTM with series decoder

    Predicts values for each timepoint in the sequence

    Args:
        input_dim: LSTM output dimension (e.g., 256)
        num_timepoints: Number of timepoints (e.g., 30)
        num_targets: Number of regression targets per timepoint (e.g., 7 emotions)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_timepoints: int = 30,
        num_targets: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_timepoints = num_timepoints
        self.num_targets = num_targets

        # MLP to expand from hidden state to series
        self.expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_timepoints * num_targets)
        )

        print(f"LSTM Series Regression Head initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output: {num_timepoints} timepoints × {num_targets} targets = {num_timepoints * num_targets}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, input_dim) - LSTM encoder output

        Returns:
            (batch, num_timepoints, num_targets, 1) - predicted values for each timepoint
        """
        B = x.shape[0]

        # Expand to series
        output = self.expand(x)  # (B, T*E)

        # Reshape to (B, T, E)
        output = output.view(B, self.num_timepoints, self.num_targets)

        # Add dummy dimension for compatibility: (B, T, E) -> (B, T, E, 1)
        output = output.unsqueeze(-1)

        return output


# For compatibility with load_model
def build_lstm_regression_head(**kwargs):
    """Factory function for LSTM regression head"""
    return LSTMRegressionHead(**kwargs)


def build_lstm_series_regression_head(**kwargs):
    """Factory function for LSTM series regression head"""
    return LSTMSeriesRegressionHead(**kwargs)


if __name__ == "__main__":
    # Test the decoders
    print("Testing LSTM Decoders...")

    # Test single-target decoder
    print("\n1. Testing LSTMRegressionHead...")
    decoder1 = LSTMRegressionHead(input_dim=256, num_targets=7, hidden_dim=128)
    x1 = torch.randn(8, 256)  # (batch=8, hidden_dim=256)
    output1 = decoder1(x1)
    print(f"   Input shape: {x1.shape}")
    print(f"   Output shape: {output1.shape}")
    assert output1.shape == (8, 7, 1), f"Expected (8, 7, 1), got {output1.shape}"

    # Test series decoder
    print("\n2. Testing LSTMSeriesRegressionHead...")
    decoder2 = LSTMSeriesRegressionHead(input_dim=256, num_timepoints=30, num_targets=7, hidden_dim=128)
    x2 = torch.randn(8, 256)  # (batch=8, hidden_dim=256)
    output2 = decoder2(x2)
    print(f"   Input shape: {x2.shape}")
    print(f"   Output shape: {output2.shape}")
    assert output2.shape == (8, 30, 7, 1), f"Expected (8, 30, 7, 1), got {output2.shape}"

    print("\n✓ All tests passed!")
