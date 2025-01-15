from typing import Optional

import torch.nn as nn

from module.models.decoder.backend.adapter import SeriesClassificationOutputAdapter, TrainableQueryProvider
from module.models.decoder.backend.modules import PerceiverDecoder

class SeriesDecoder(nn.Module):
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
                 num_output_queries: int = 20, # set to timesteps
                 num_output_query_channels: int = 256,
                 num_classes: int = 100,
                 # Series specific
                 num_targets: int = 7 # e.g. num of emotions
                 ):
        super().__init__()

        # query for num_classes, num_output_queries-times (i.e. timestamps)
        output_query_provider = TrainableQueryProvider(
            num_queries=num_output_queries,
            num_query_channels=num_output_query_channels,
            init_scale=init_scale,
        )

        output_adapter = SeriesClassificationOutputAdapter(
            num_classes=num_classes,
            num_output_query_channels=num_output_query_channels,
            num_targets=num_targets
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
        output = self.decoder(x) # expecting x of shape (batch_size, num_latents, num_latent_channels)
        return output
