import sys
sys.path.append('../')

import argparse
from pathlib import Path

import numpy as np
import torch
import os
from utils.pickle_model import load_model

from dataset.omsignal_dataset import OMSignalDataset
from torchvision.transforms import Compose
from dataset.preprocessor import Preprocessor
from dataset.segmenter import PairedHeartBeatSegmenter,HeartBeatSegmenter
from models.multi_task_model import MultiTaskNet
from models.multi_task_predictions import multi_task_predict
from utils.memfile_utils import write_memfile


def eval_model(dataset_file, model_filename):
    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''


    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading model from", model_filename.absolute())
        model = torch.load(model_filename, map_location=device)

    if model:

        '''
        Multitask prediction

        '''

        model = MultiTaskNet(230, 1, 35, 30)
        model.load_state_dict(torch.load(model_filename))

        test_transforms = Compose([Preprocessor(),
                                   PairedHeartBeatSegmenter(230)
                                   ])

        test_data = OMSignalDataset(False, True, dataset_file, test_transforms)

        y_pred, true_labels = multi_task_predict(model, test_data)


        '''
        ID and RR prediction with PCA + LDA and PCA + SVR

        '''

        ID_PCA = load_model("/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt3/model/ID_PCA.pkl")
        RR_PCA = load_model("/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt3/model/RR_PCA.pkl")
        ID_model = load_model("/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt3/model/ID_model.pkl")
        RR_model = load_model("/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt3/model/RR_model.pkl")

        transform_ID = Compose([Preprocessor(),
                                HeartBeatSegmenter(110, take_average=True)])

        transform_RR = Compose([Preprocessor(),
                                PairedHeartBeatSegmenter(230, take_average=True)])

        ID_data = OMSignalDataset(False, True, dataset_file, transform_ID)
        RR_data = OMSignalDataset(False, True, dataset_file, transform_RR)

        test_ID_x = np.empty((160, 110))
        test_RR_x = np.empty((160, 230))
        test_y = np.empty((160, 4))

        for i in range(160):
            ecg, target = ID_data[i]
            test_ID_x[i, :] = np.array(ecg)

            ecg2, target2 = RR_data[i]
            test_RR_x[i, :] = np.array(ecg2)

            test_y[i] = target2

        pred_ID = ID_PCA.transform(test_ID_x)
        pred_ID = ID_model.predict(pred_ID)

        pred_RR = RR_PCA.transform(test_RR_x)
        pred_RR = RR_model.predict(pred_RR)

        y_pred[:, 3] = pred_ID
        y_pred[:, 2] = pred_RR

    else:

        print("\nYou did not specify a model, generating dummy data instead!")
        n_classes = 32
        num_data = 10

        y_pred = np.concatenate(
            [np.random.rand(num_data, 3),
             np.random.randint(0, n_classes, (num_data, 1))
             ], axis=1
        ).astype(np.float32)

    return y_pred

if __name__ == "__main__":
    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='')
    # dataset_dir will be the absolute path to the dataset to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    dataset_file = args.dataset
    results_dir = args.results_dir
    #########################################

    ###### MODIFY THIS SECTION ######
    # Put your group name here
    group_name = "b2pomt3"

    model_filename = '/rap/jvb-000-aa/COURS2019/etudiants/submissions/'
    model_filename = os.path.join(model_filename, group_name, 'model', 'multi_task_net.pt')
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    #################################

    ###### DO NOT MODIFY THIS SECTION ######
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_file, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 2, "Make sure ndim=2 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    write_memfile(y_pred, results_fname)
    #########################################
