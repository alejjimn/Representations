import torch


class ToFloatTensorTransform(object):

    def __init__(self):
        print("")

    def __call__(self, sample):
        """
        Parameters
        ---------------
        sample : numpy.ndarray

        Returns
        ---------------
        torch.Tensor type float
        """
        sample = torch.Tensor(sample).float()
        return torch.unsqueeze(sample, 0)
