import torch
from einops import rearrange
from torch import nn as nn

class OutputAdapter(nn.Module):
    """Transforms generic decoder cross-attention output to task-specific output."""

class SeriesClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_query_channels: int,
        num_targets: int
    ):
        super().__init__()
        self.num_targets = num_targets
        self.linear = nn.Linear(num_output_query_channels, num_classes*num_targets)

    def forward(self, x):
        x = self.linear(x).squeeze(dim=1)
        x = rearrange(x, 'b t (ta c) -> b t ta c', ta=self.num_targets) # (batch_size, time_sequence, num_targets, num_classes)
        return x if x.shape[-1] > 1 else x.squeeze() # (batch_size, time_sequence, num_targets) for regression

class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_query_channels: int,
    ):
        super().__init__()
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)

class QueryProvider:
    """Provider of cross-attention query input."""

    @property
    def num_query_channels(self):
        raise NotImplementedError()

    def __call__(self, x=None):
        raise NotImplementedError()

class TrainableQueryProvider(nn.Module, QueryProvider):
    """Provider of learnable cross-attention query input.

    This is the latent array in Perceiver IO encoders and the output query array in most Perceiver IO decoders.
    """

    def __init__(self, num_queries: int, num_query_channels: int, init_scale: float = 0.02):
        super().__init__()
        self._query = nn.Parameter(torch.empty(num_queries, num_query_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._query.normal_(0.0, init_scale)

    @property
    def num_query_channels(self):
        return self._query.shape[-1]

    def forward(self, x=None):
        return rearrange(self._query, "... -> 1 ...")