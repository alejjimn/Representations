import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.omsignal_dataset import OMSignalDataset
#from dataset.unlabeled_dataset import UnlabeledOMSignalDataset
from torchvision.transforms import Compose

from dataset.preprocessor import Preprocessor
from dataset.segmenter import PairedHeartBeatSegmenter,HeartBeatSegmenter
from dataset.to_float_tensor_transform import ToFloatTensorTransform
from dataset.encode_transform import Encode
from models.autoencoder import AutoEncoder_2l,AutoEncoder_3l,AutoEncoder_3l_v2


from models.multi_task_model import MultiTaskNet
from models.multi_task_trainer import MultiTaskTrainer

from utils.dataset_utils import generate_transformed_dataset
from utils.ids_conversion import convert_back_ids,convert_ids

from models.multi_task_predictions import multi_task_predict

from scipy.stats import kendalltau, mode
from sklearn.metrics import accuracy_score

TRAIN_PSEUDOLABELED_PATH ="/rap/jvb-000-aa/COURS2019/etudiants/submissions/b3pomt2/datasets/pseudo_labels/PseudoLabeledData.dat"
TRAIN_UNLABELED_PATH = "/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/MILA_UnlabeledData.dat"
TRAIN_LABELED_PATH = "/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/MILA_TrainLabeledData.dat"
VALID_LABELED_PATH = "/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/MILA_ValidationLabeledData.dat"

SEG_TRAIN_UNLABELED = "results/OMSignal_SegmentedData_Backup_2.dat"
SEG_TRAIN_LABELED = "results/OMSignal_Segmented_TrainLabeledData_encode.dat"
SEG_VALID_LABELED = "results/OMSignal_Segmented_ValidationLabeledData_encode.dat"

SAVE_PATH = "./results/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#encoder = encoder = AutoEncoder_3l_v2(1, hidden_size=[16,32,64], kernel_size=[13,9,5,5,10,13], padding=[6,4,2,2,4,6], stride=[1,1,1,1,1,1])
#encoder.load_state_dict(torch.load("../checkpoint/AutoEncoder_21_Adam_train1.tar", map_location=DEVICE)["model_state_dict"])
origin_dataset_path = TRAIN_PSEUDOLABELED_PATH
destination_dataset_path = SEG_TRAIN_LABELED
#transforms = Compose([Preprocessor(), HeartBeatSegmenter(230)])
transforms = Compose([Preprocessor(), PairedHeartBeatSegmenter(230)])
#transforms = Compose([Preprocessor()])
generate_transformed_dataset(True,
                                 origin_dataset_path,
                                 destination_dataset_path,
                                 transforms)

origin_dataset_path = VALID_LABELED_PATH
destination_dataset_path = SEG_VALID_LABELED
#transforms = Compose([Preprocessor(), HeartBeatSegmenter(230)])
transforms = Compose([Preprocessor(), PairedHeartBeatSegmenter(230)])

generate_transformed_dataset(True,
                                 origin_dataset_path,
                                 destination_dataset_path,
                                 transforms)

transforms_train = Compose([ToFloatTensorTransform()])
transforms_valid =  Compose([ToFloatTensorTransform()])

train_labeled_data = OMSignalDataset(True,True,SEG_TRAIN_LABELED,transforms_train)
valid_labeled_data = OMSignalDataset(True,True,SEG_VALID_LABELED,transforms_valid)
#train_labeled_data = OMSignalDataset(False,True,TRAIN_LABELED_PATH,transforms_train)
#valid_labeled_data = OMSignalDataset(False,True,VALID_LABELED_PATH,transforms_valid)

trainloader = torch.utils.data.DataLoader(train_labeled_data, batch_size=16,
                                          shuffle=True)

validloader = torch.utils.data.DataLoader(valid_labeled_data , batch_size=16,
                                         shuffle=False)


print("Starting process.....")
model = MultiTaskNet(230, 1, 35, 30)
Trainer = MultiTaskTrainer(model,
                           trainloader,
                           validloader,
                           SAVE_PATH,
                           learning_rate=0.0001,
                           batch_size=16,
                           weight_decay=0.01,
                           log=True,
                           seed=111)


'''
train model
'''
print("Training model.....")
Trainer.train(50)
'''
predict

'''

#model = MultiTaskNet(224, 1, 35, 30)
model = MultiTaskNet(230, 1, 35, 30)
model.load_state_dict(torch.load('./results/multi_task_net.pt'))

test_transforms = Compose([Preprocessor(),
                           PairedHeartBeatSegmenter(230)
                           ])

#test_transforms = Compose([ToFloatTensorTransform(),Preprocessor()])
print("Testing model.....")
test_data = OMSignalDataset(False, True, VALID_LABELED_PATH, test_transforms)

y_pred,true_labels = multi_task_predict(model,test_data)


'''
print Metrics

'''

converted_ids = convert_ids(true_labels[:,3])

print("ID: " + str(accuracy_score(converted_ids,y_pred[:,3])))
print("PR: " + str(kendalltau(true_labels[:,0],y_pred[:,0])[0]))
print("RT: " + str(kendalltau(true_labels[:,1],y_pred[:,1])[0]))
print("RR: " + str(kendalltau(true_labels[:,2],y_pred[:,2])[0]))