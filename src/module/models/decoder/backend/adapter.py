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
        num_targets: int,
        downstream_task_type: str = 'regression'
    ):
        super().__init__()
        self.num_targets = num_targets
        self.downstream_task_type = downstream_task_type
        self.linear = nn.Linear(num_output_query_channels, num_classes*num_targets)

    def forward(self, x):
        x = self.linear(x).squeeze(dim=1)
        x = rearrange(x, 'b t (ta c) -> b t ta c', ta=self.num_targets) # (batch_size, time_sequence, num_targets, num_classes)
        # For regression, squeeze out the num_classes=1 dimension
        # For classification, keep the num_classes dimension (even if it's 1 for binary classification)
        if self.downstream_task_type == 'regression' and x.shape[-1] == 1:
            return x.squeeze() # (batch_size, time_sequence, num_targets) for regression
        else:
            return x # (batch_size, time_sequence, num_targets, num_classes) for classification

class AveragedSeriesOutputAdapter(OutputAdapter):
    """Adapter that averages series outputs over time for single target prediction (e.g., Sex).

    Takes series decoder output and averages over the time dimension to produce
    a single prediction per batch item.
    """
    def __init__(
        self,
        num_classes: int,
        num_output_query_channels: int,
        num_targets: int = 1,  # For sex classification, typically 1 target
        downstream_task_type: str = 'classification'
    ):
        super().__init__()
        self.num_targets = num_targets
        self.downstream_task_type = downstream_task_type
        self.linear = nn.Linear(num_output_query_channels, num_classes * num_targets)

    def forward(self, x):
        # x shape: (batch_size, num_queries, num_output_query_channels)
        x = self.linear(x)  # (batch_size, num_queries, num_classes*num_targets)
        x = rearrange(x, 'b t (ta c) -> b t ta c', ta=self.num_targets)  # (batch, time, targets, classes)

        # Average over time dimension
        x = x.mean(dim=1)  # (batch, targets, classes)

        # For single target (like sex), squeeze the target dimension
        if self.num_targets == 1:
            x = x.squeeze(dim=1)  # (batch, classes)

        return x

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