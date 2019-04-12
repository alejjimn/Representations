import torch


class Encode(object):
    def __init__(self, model):
        """Returns sample as output of encoder
        Args:
            model (nn.Module): trained model with a __call__ function
        Returns:
            encoded_sample (torch.tensor): sample encoded
        """
        self.model = model

    def __call__(self, sample):

        with torch.no_grad():
            sample = sample.unsqueeze(0)
            encoded_sample, _ = self.model(sample)
        return encoded_sample
