import torch
import torch.nn.functional as F


class Preprocessor(object):
    """Preprocess signal with moving window.

    Args:
        ma_window_size: (in seconds) window size to use
                        for moving average baseline wander removal
        mv_window_size: (in seconds) window size to use
                        for moving average RMS normalization
        num_samples_per_second: (Hertz)
    """
    def __init__(
            self,
            ma_window_size=2,
            mv_window_size=4,
            num_samples_per_second=125):

        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd

        self.maKernelSize = (ma_window_size * num_samples_per_second) + 1

        # Kernel size to use for moving average normalization: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd

        self.mvKernelSize = (mv_window_size * num_samples_per_second) + 1

    def __call__(self, sample):
        x = torch.tensor(sample)
        x = x.view(1, 1, -1)

        # Remove window mean and standard deviation

        x = (x - torch.mean(x, dim=2, keepdim=True)) / \
            (torch.std(x, dim=2, keepdim=True) + 0.00001)

        # Moving average baseline wander removal

        x = x - F.avg_pool1d(
            x, kernel_size=self.maKernelSize,
            stride=1, padding=(self.maKernelSize - 1) // 2
        )

        # Moving RMS normalization

        x = x / (
            torch.sqrt(
                F.avg_pool1d(
                    torch.pow(x, 2),
                    kernel_size=self.mvKernelSize,
                    stride=1, padding=(self.mvKernelSize - 1) // 2
                )) + 0.00001
        )

        # Don't backpropagate further
        return x.squeeze(0)
