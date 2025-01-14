from typing import Optional

import torch.nn as nn

from module.models.decoder.backend.adapter import ClassificationOutputAdapter, TrainableQueryProvider
from module.models.decoder.backend.modules import PerceiverDecoder

class SingleTargetDecoder(nn.Module):
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
                 num_output_queries: int = 1,
                 num_output_query_channels: int = 256,
                 num_classes: int = 100 # set to 1 for regression
                 ):
        super().__init__()
        
        output_query_provider = TrainableQueryProvider(
            num_queries=1,
            num_query_channels=num_output_query_channels,
            init_scale=init_scale,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=num_classes,
            num_output_query_channels=num_output_query_channels,
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
            dropout=dropout
        )

    def forward(self, x):
        return self.decoder(x) # x expected to be a tensor of shape (batch_size, num_latents, num_latent_channels)