from torch import nn
import torch


class AutoformerSeriesDecomposition(nn.Module):
    """
    Series Decomposition block, to highlight the trend of time series:

        x_trend = AvgPool(Padding(X))
        x_seasonal = X - x_trend
    """
    def __init__(self, kernel_size):
        super(AutoformerSeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # pooling
        x = self.avg(x.permute(0, 2, 1))
        x_trend = x.permute(0, 2, 1)

        # calculate x_seasonal
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


class AutoformerLayernorm(nn.Module):
    """
    Special designed layer normalization for the seasonal part, calculated as:
    AutoformerLayernorm(x) = nn.LayerNorm(x) - torch.mean(nn.LayerNorm(x))
    """
    def __init__(self, channels):
        super(AutoformerLayernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


