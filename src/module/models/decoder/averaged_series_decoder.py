from typing import Optional

import torch.nn as nn

from .backend.adapter import AveragedSeriesOutputAdapter, TrainableQueryProvider
from .backend.modules import PerceiverDecoder

class AveragedSeriesDecoder(nn.Module):
    """Series decoder that averages predictions over time for single target tasks (e.g., Sex classification).

    This decoder processes temporal sequences and averages the predictions across timesteps
    to produce a single prediction per batch item. Useful for subject-level predictions
    where we want to aggregate information across the entire temporal sequence.

    For a 20-frame sequence predicting Sex (binary):
        - Input: (batch_size, num_latents, num_latent_channels)
        - Internal: (batch_size, 20_timesteps, num_classes)
        - Output: (batch_size, num_classes) after averaging over 20 timesteps
    """
    def __init__(self,
                 # PerceiverIO specific
                 num_latents: int,
                 num_latent_channels: int,
                 activation_checkpointing: bool = False,
                 # Decoder specific
                 activation_offloading: bool = False,
                 num_cross_attention_heads: int = 8,
                 num_cross_attention_qk_channels: Optional[int] = None,
                 num_cross_attention_v_channels: Optional[int] = None,
                 cross_attention_widening_factor: int = 1,
                 cross_attention_residual: bool = True,
                 dropout: float = 0.0,
                 init_scale: float = 0.02,
                 # Classification specific
                 num_output_queries: int = 20,  # set to timesteps (e.g., 20 for seq_length=20)
                 num_output_query_channels: int = 256,
                 num_classes: int = 2,  # e.g., 2 for binary sex classification
                 # Target specific
                 num_targets: int = 1,  # For sex, age, etc. - single target
                 # Task type
                 downstream_task_type: str = 'classification'
                 ):
        super().__init__()

        # Query for num_classes, num_output_queries-times (i.e. timestamps)
        output_query_provider = TrainableQueryProvider(
            num_queries=num_output_queries,
            num_query_channels=num_output_query_channels,
            init_scale=init_scale,
        )

        # Use the averaged adapter to collapse time dimension
        output_adapter = AveragedSeriesOutputAdapter(
            num_classes=num_classes,
            num_output_query_channels=num_output_query_channels,
            num_targets=num_targets,
            downstream_task_type=downstream_task_type
        )

        self.decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=num_latent_channels,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
            num_cross_attention_heads=num_cross_attention_heads,
            num_cross_attention_qk_channels=num_cross_attention_qk_channels,
            num_cross_attention_v_channels=num_cross_attention_v_channels,
            cross_attention_widening_factor=cross_attention_widening_factor,
            cross_attention_residual=cross_attention_residual,
            dropout=dropout,
        )

    def forward(self, x):
        # x expected to be (batch_size, num_latents, num_latent_channels)
        output = self.decoder(x)
        # output will be (batch_size, num_classes) after averaging in adapter
        return output
