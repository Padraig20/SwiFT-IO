"""
LSTM Encoder for fMRI Emotion Decoding

A simple LSTM-based baseline for comparison with spatiotemporal transformer models.
This encoder processes temporal sequences of fMRI data.

Author: For comparison with SwiFT-IO
Date: 2025-10-14
"""

import torch
import torch.nn as nn
from einops import rearrange


class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder for fMRI sequence encoding

    Processes 4D fMRI sequences (batch, channel, height, width, depth, time) through:
    1. Spatial flattening/pooling to reduce dimensionality
    2. LSTM to capture temporal dynamics
    3. Output the final hidden state or all hidden states

    Args:
        input_dim: Flattened spatial feature dimension (default: 96*96*96 = 884736)
        hidden_dim: LSTM hidden state dimension (default: 256)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate between LSTM layers (default: 0.3)
        bidirectional: Whether to use bidirectional LSTM (default: False)
        pooling_type: How to reduce spatial dims ['flatten', 'avgpool', 'adaptive'] (default: 'adaptive')
        pooled_spatial_dim: Target spatial dimension after pooling (default: 32)
        return_sequence: Whether to return all hidden states or just last (default: False)
    """

    def __init__(
        self,
        input_dim: int = 884736,  # 96^3
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling_type: str = 'adaptive',  # 'flatten', 'avgpool', 'adaptive'
        pooled_spatial_dim: int = 32,
        return_sequence: bool = False,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling_type = pooling_type
        self.pooled_spatial_dim = pooled_spatial_dim
        self.return_sequence = return_sequence

        # Spatial dimension reduction
        if pooling_type == 'adaptive':
            # Adaptive pooling to fixed size (much more efficient than flatten)
            self.spatial_reduction = nn.Sequential(
                nn.AdaptiveAvgPool3d(pooled_spatial_dim),
                nn.Flatten(start_dim=1)  # (B, C, H, W, D) -> (B, C*H*W*D)
            )
            lstm_input_dim = pooled_spatial_dim ** 3  # e.g., 32^3 = 32768

        elif pooling_type == 'avgpool':
            # Average pooling with specific kernel size
            pool_size = 96 // pooled_spatial_dim
            self.spatial_reduction = nn.Sequential(
                nn.AvgPool3d(kernel_size=pool_size, stride=pool_size),
                nn.Flatten(start_dim=1)
            )
            lstm_input_dim = pooled_spatial_dim ** 3

        elif pooling_type == 'flatten':
            # Direct flattening (memory intensive!)
            self.spatial_reduction = nn.Flatten(start_dim=1)
            lstm_input_dim = input_dim
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

        # Optional projection layer to further reduce dimension
        self.input_projection = nn.Linear(lstm_input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

        print(f"LSTM Encoder initialized:")
        print(f"  Pooling: {pooling_type} -> {pooled_spatial_dim}^3 = {pooled_spatial_dim**3}")
        print(f"  LSTM input dim: {hidden_dim}")
        print(f"  LSTM hidden dim: {hidden_dim}")
        print(f"  LSTM layers: {num_layers}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Output dim: {self.output_dim}")

    def forward(self, x):
        """
        Forward pass through LSTM encoder

        Args:
            x: Input tensor of shape (batch, channel, height, width, depth, time)
               e.g., (8, 1, 96, 96, 96, 30)

        Returns:
            If return_sequence=False: (batch, output_dim) - last hidden state
            If return_sequence=True: (batch, time, output_dim) - all hidden states
        """
        B, C, H, W, D, T = x.shape

        # Rearrange to process each timepoint
        # (B, C, H, W, D, T) -> (B, T, C, H, W, D)
        x = rearrange(x, 'b c h w d t -> b t c h w d')

        # Process each timepoint through spatial reduction
        x = rearrange(x, 'b t c h w d -> (b t) c h w d')
        x = self.spatial_reduction(x)  # (B*T, lstm_input_dim)

        # Ensure dtype consistency (convert to float32 if needed)
        x = x.float()

        # Project to hidden dimension
        x = self.input_projection(x)  # (B*T, hidden_dim)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Reshape back to sequence
        x = rearrange(x, '(b t) d -> b t d', b=B, t=T)  # (B, T, hidden_dim)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, T, hidden_dim * num_directions)

        if self.return_sequence:
            # Return all hidden states
            return lstm_out  # (B, T, output_dim)
        else:
            # Return only last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim * 2)
            else:
                h_n = h_n[-1]  # (B, hidden_dim)

            return h_n  # (B, output_dim)


class LSTMEncoderLight(nn.Module):
    """
    Lightweight LSTM encoder with much smaller spatial dimension

    More efficient version for quick experiments
    Uses aggressive spatial pooling (e.g., 8x8x8) before LSTM
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooled_spatial_dim: int = 8,
        **kwargs
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling_type='adaptive',
            pooled_spatial_dim=pooled_spatial_dim,
            bidirectional=False,
            return_sequence=False,
            **kwargs
        )

    def forward(self, x):
        return self.encoder(x)


# For compatibility with load_model
def build_lstm_encoder(**kwargs):
    """Factory function for LSTM encoder"""
    return LSTMEncoder(**kwargs)


def build_lstm_encoder_light(**kwargs):
    """Factory function for lightweight LSTM encoder"""
    return LSTMEncoderLight(**kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM Encoder...")

    model = LSTMEncoder(
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pooling_type='adaptive',
        pooled_spatial_dim=16
    )

    # Simulated input
    x = torch.randn(2, 1, 96, 96, 96, 30)  # (batch=2, channel=1, H=96, W=96, D=96, time=30)

    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    print("\n✓ Test passed!")
