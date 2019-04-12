# For Project Cours Winter 2019
# OM Signal Project Block 2
import sys
import pandas as pd
import torch.utils.data as data
import numpy as np
sys.path.append('../')
from utils.file_utils import read_data
from utils.ids_conversion import convert_ids


class OMSignalDataset(data.Dataset):
    """ OMSignal dataset """
    def __init__(self, useSegmentedData, useLabeledData,
                 filepath, transform=None):
        """
        Parameters
        ----------
        useSegmentedData : boolean
            if True, this dataset is for segmented data (dimension 230)
        useLabeledData : boolean
            if True, this dataset is for labeled data (dimension 3754)
        filepath : string
           dataset location
        transform : torchvision.transforms.transforms.Compose
            composition of all the transformations needed on the dataset
        """
        np.set_printoptions(threshold=sys.maxsize)

        self.useLabeledData = useLabeledData
        self.useSegmentedData = useSegmentedData

        if self.useLabeledData:
            self.shape = (160, 3754)
        else:
            self.shape = (657233, 3750)

        self.transform = transform
        self.data = None

        if useSegmentedData:
            # if segmented, load text file using panda (to speed up the process)
            self.data = pd.read_csv(filepath, delimiter=" ").values
            if self.useLabeledData:
                # convert ids to smooth interval 0 to 31
                self.data[:, -1] = convert_ids(self.data[:, -1])
        else:
            # read memmap for non-segmented
            self.data = read_data(filepath, self.shape)
            

    def __len__(self):
        """Get the number of ecgs in the dataset.
        Returns
        -------
        int
           The number of ecg in the dataset.
        """
        return len(self.data)

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
        current_item = self.data[index]

        if self.useLabeledData:
            targets = current_item[-4:]
            current_item = current_item[:-4]
        else:
            targets = []

        # Note : transform for labeled data should include data augmentation
        # and preprocessing. transform for unlabeled data should add a transform
        # to get the target
        if self.transform is not None:
            transformed_item = self.transform(current_item)
            return transformed_item, targets

        return current_item, targets
