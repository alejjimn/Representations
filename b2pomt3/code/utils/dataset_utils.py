import numpy as np
from dataset.omsignal_dataset import OMSignalDataset
import random
from dataset.to_float_tensor_transform import ToFloatTensorTransform


def generate_transformed_dataset(use_labeled_data, origin_dataset_path,
                                 destination_dataset_path, offline_transforms,
                                 encode=None):
    """
    From a labeled or unlabeled dataset, this method will segment all 30s
    sample of ECGs window into samples containing 1 RR interval
    Parameters
    ----------
     useLabeledData : boolean
            if True, this dataset is for labeled data (dimension 3754)
    origin_dataset_path : string
       the path to the dataset we want to apply the segmentation
    destination_dataset_path : string
       where to save the newly generated dataset
    offline_transforms : torchvision.transforms.transforms.Compose
        all the transformations we want to apply offline
    encode : int
    """
    unlabeled_data = OMSignalDataset(False, use_labeled_data,
                                     origin_dataset_path,
                                     offline_transforms)
    to_float = ToFloatTensorTransform()

    with open(destination_dataset_path, 'w') as the_file:
        num_segments = 0
        for sample, target in unlabeled_data:
            # limit number of segments in file
            if num_segments >= 1000000:
                break

            for segmented_sample in sample:
                num_segments = num_segments+1

                if encode is not None:
                    segmented_sample = to_float(segmented_sample)
                    segmented_sample = encode(segmented_sample)
                    segmented_sample = segmented_sample.squeeze(0)

                segment = np.concatenate((segmented_sample, target))
                # convert segment values to string
                str_list = [str(value) for value in list(segment)]
                # write line
                the_file.write(' '.join(str_list) + "\n")


def generate_random_dataset(use_labeled_data,
                            origin_dataset_path,
                            destination_dataset_path,
                            offline_transforms,
                            nsegments=8,
                            number_of_average=1000):
    """
    From a labeled or unlabeled dataset, this method will segment all 30s
    sample of ECGs window into samples containing 1 RR interval and make
    multiple random average of those segment

    Parameters
    ----------
     useLabeledData : boolean
            if True, this dataset is for labeled data (dimension 3754)
    origin_dataset_path : string
       the path to the dataset we want to apply the segmentation
    destination_dataset_path : string
       where to save the newly generated dataset
    offline_transforms : torchvision.transforms.transforms.Compose
        all the transformations we want to apply offline
    nsegments : int
                number of segments to average
    number_of_average :int
                number of random average to repeat
    """

    unlabeled_data = OMSignalDataset(False, use_labeled_data,
                                     origin_dataset_path,
                                     offline_transforms)
    with open(destination_dataset_path, 'w') as the_file:
        num_segments = 0
        for sample, target in unlabeled_data:
            for i in range(number_of_average):
                random_index = []
                if len(sample) >= nsegments:
                    random_index = random.sample(set(range(len(sample))), nsegments)
                else:
                    random_index = range(len(sample))

                random_segments = np.array([sample[index] for index in random_index])

                average_segment = np.mean(random_segments, axis=0)

                num_segments = num_segments + 1

                segment = np.concatenate((average_segment, target))
                # convert segment values to string
                str_list = [str(value) for value in list(segment)]

                # write line
                the_file.write(' '.join(str_list) + "\n")
