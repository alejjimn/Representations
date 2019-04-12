# For Project Cours Winter 2019
# OM Signal Project Block 2
import sys
import torch
import torch.utils.data as data
import numpy as np
sys.path.append('../')

import data_augmentation as da


def read_data(filename):
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype='float32', mode='r', shape=(657233, 3750))
    unlabeled_data = np.zeros(shape=(657233, 3750))
    unlabeled_data[:] = fp[:]
    del fp

    return unlabeled_data


class UnlabeledOMSignalDataset(data.Dataset):
    """ SVHN dataset """

    def __init__(self, filepath, use_transform=False):
        """
        Parameters
        ----------
        filepath : string
           dataset location
        use_transform : boolean
           whether or not we are using the transformation
        """
        self.use_transform = use_transform
        self.ecg = read_data(filepath)

    def __len__(self):
        """Get the number of ecgs in the dataset.
        Returns
        -------
        int
           The number of ecg in the dataset.
        """
        return len(self.ecg)

    def __getitem__(self, index):
        """Get the items : ecg, target (userid by default)
        Parameters
        ----------
        index : int
           Index
        Returns
        -------
        img : tensor
           The ecg
        target : int or float, or a tuple
           When int, it is the class_index of the target class.
           When float, it is the value for regression
        """
        ecg = self.ecg[index]

        if self.use_transform:
            ecg = self.unlabeled_transform(ecg)
        else:
            ecg = torch.Tensor(ecg).float()

        return ecg.unsqueeze(0)

    def unlabeled_transform(self, x):
        """
        Take an ECG numpy array x
        Randomly flip it, shift it and noise it
        """
        # first, random flip
        if np.random.random() > 0.5:
            x = da.upside_down_inversion(x)
        # shift the series by 1 to 25 steps
        if np.random.random() > 0.5:
            x = da.shift_series(x, shift=np.random.randint(1, 26))
        # add partial gaussian noise 50% of the time
        if np.random.random() > 0.5:
            x = da.adding_partial_noise(
                x, second=np.random.randint(0, 29),
                duration=np.random.randint(1, 3)
            )
        return torch.Tensor(x).float()
