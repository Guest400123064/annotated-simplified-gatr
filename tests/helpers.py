import torch
from tests.config import BasicConfig

def generate_random_data(config: BasicConfig) -> torch.Tensor:
    r"""Generate random data points for testing.

    Parameters
    ----------
    config : BasicConfig
        Configuration object containing parameters for generating random data.

    Returns
    -------
    torch.Tensor
        Generated random data of shape (batch_size, num_points, input_channels, 16).
    """
    batch_size = int(torch.randint(1, config.max_batch_size + 1, (1,)).item())
    num_points = int(torch.randint(1, config.max_context_size + 1, (1,)).item())
    input_channels = int(torch.randint(1, config.max_channel_size + 1, (1,)).item())
    return torch.randn(batch_size, num_points, input_channels, 16)
