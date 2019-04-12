import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append('../')


from torchvision.transforms import Compose
from torch.utils.data.sampler import SubsetRandomSampler
from utils.arguments_parser import ArgumentsParser
from utils.config_file_accessor import ConfigurationFileLoader
from dataset.omsignal_dataset import OMSignalDataset
from dataset.to_float_tensor_transform import ToFloatTensorTransform
from models.autoencoder import AutoEncoder_2l, AutoEncoder_3l, AutoEncoder_3l_v2, weights_init

# global constant
HERE = os.path.dirname(os.path.abspath(__file__))  # this file's location
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_autoencoder(net, criterion, optimizer, n_epochs, trainloader,
                      validloader, checkpoint, outdir, file_suffix):
    """
    Model training for Autoencoder

        Args:
            net (torch.nn.Module): autoencoder model to train
            criterion (torch.nn.modules.loss): loss function to minimize
            optimizer (torch.optim): algorithm for minimizing the loss function
            n_epochs (int): maximum number of epochs to train for
            trainloader (torch.utils.data.DataLoader): dataloader generating training samples
            validloader (torch.utils.data.DataLoader): dataloader generating validation samples
            checkpoint (str, path-like, NoneType): path to the training checkpoint
            outdir (path-like, str): path to directory to save training checkpoints
            file_suffix (str): suffix to name checkpoint file
        Returns:
            train_history (list): history of training losses
            valid_history (list): history of validation losses
            best_valid_loss (int): best validation loss
    """
    train_history = []
    valid_history = []

    start_epoch = 0

    if checkpoint is None:
        best_model = net.state_dict()
        best_valid_loss = math.inf
    else:
        # If checkpoint exists, start from last checkpoint and load checkpoint state.
        state = torch.load(checkpoint, map_location=DEVICE)
        net.load_state_dict(state["model_state_dict"])
        best_model = net.state_dict()
        optimizer.load_state_dict(state["optimizer_state_dict"])
        best_valid_loss = state["best_valid_loss"]
        start_epoch = state["epoch"] + 1
        print("Loaded model", checkpoint, "at epoch",
              state["epoch"], "with loss of", state["loss"],
              "and best validation loss of", best_valid_loss)

    # Training model
    try:
        for epoch in range(start_epoch, n_epochs):
            print("-" * 25)
            print("EPOCH", epoch+1)
            print("[Epoch, batch index]:  loss")

            for phase in ["train", "valid"]:
                if phase == "train":
                    print("Start Training Phase.")
                    dataloader = trainloader
                    net.train()
                else:
                    print("Start Validation Phase.")
                    dataloader = validloader
                    net.eval()

                running_loss = 0.0
                epoch_loss = 0.0
                for idx, (inputs, _) in enumerate(dataloader):
                    # set the inputs and targets
                    inputs = inputs.to(DEVICE)
                    targets = inputs

                    # Reset the gradients
                    optimizer.zero_grad()

                    # forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        _, outputs = net(inputs)
                        loss = criterion(outputs, targets)

                    # Back pass
                    if phase == "train":
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                        optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    if idx % 100 == 99:
                        print("[%d, %5d]  loss: %.8f" %
                              (epoch+1, idx+1,  running_loss / 100))
                        running_loss = 0.0

                epoch_loss = epoch_loss / len(dataloader)
                print("{} loss: {:.4f}".format(phase, epoch_loss))

                # Keep track of history for plotting of learning curves
                if phase == "train":
                    train_history.append(epoch_loss)
                if phase == "valid":
                    valid_history.append(epoch_loss)

                # Keep track of the best model
                if phase == 'valid' and epoch_loss < best_valid_loss:
                    print("New best model found!")
                    print("New record loss: {}, previous record loss: {}".format(epoch_loss, best_valid_loss))
                    best_valid_loss = epoch_loss
                    best_model = net.state_dict()
            print("-" * 25)

    finally:
        model_path = os.path.join(  # Save model regardless of outcome
            outdir, "{}_{}_{}.tar".format(type(net).__name__,type(optimizer).__name__, file_suffix))
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "best_model": best_model,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss,
            "best_valid_loss": best_valid_loss,
            "train_history": train_history,
            "val_history": valid_history,
        }, model_path)

        return train_history, valid_history, best_valid_loss


if __name__ == '__main__':
    # parse args
    args = ArgumentsParser.get_parsed_arguments()
    config = ConfigurationFileLoader(args["config"])
    hp = config.get_hyperparameters()

    use_gpu = hp["use_gpu"]

    # Get hyperparameters
    model = hp["ae_model"]
    hidden_size = hp["ae_hidden_size"]
    kernel_size = hp["ae_kernel_size"]
    stride = hp["ae_stride"]
    padding = hp["ae_padding"]
    batch_size = hp["ae_batch_size"]
    n_epochs = hp["ae_n_epochs"]
    criterion = hp["ae_criterion"]
    optimizer = hp["ae_optimizer"]
    lr = hp["ae_lr"]
    checkpoint = hp["ae_checkpoint"]
    output_dir = hp["ae_output_dir"]
    file_suffix = hp["ae_file_suffix"]

    # check if checkpoint was given
    try:
        checkpoint = torch.load(checkpoint)
        print("Checkpoint Found.")
    except Exception:
        checkpoint = None
        print("Checkpoint Not Found.")

    # Set random seed
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Training autoencoder
    if hp["train_ae"]:
        transform = Compose([ToFloatTensorTransform()])
        dataset = OMSignalDataset(True, False, hp["segmented_unlabeled_path"],
                                  transform)

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=train_sampler)
        validloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=valid_sampler)

        if model == "AutoEncoder_2l":
            model_ = AutoEncoder_2l(1, hidden_size=hidden_size,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride)
            model_.apply(weights_init)

        if model == "AutoEncoder_3l":
            model_ = AutoEncoder_3l(1, hidden_size,
                                    kernel_size,
                                    stride,
                                    padding)
            model_.apply(weights_init)
        if model == "AutoEncoder_3l_v2":
            model_ = AutoEncoder_3l_v2(1, hidden_size=hidden_size, kernel_size=kernel_size, padding=padding, stride=stride)
            model_.apply(weights_init)

        model_.to(DEVICE)

        if optimizer == "SGD":
            o = optim.SGD(
                model_.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "Adam":
            o = optim.Adam(model_.parameters(), lr=lr)
        elif optimizer == "RMSprop":
            o = optim.RMSprop(
                model_.parameters(), lr=lr, momentum=0.9)

        if criterion == "MSELoss":
            c = nn.MSELoss()

        print("-" * 50)
        train_autoencoder(model_, c, o, n_epochs, trainloader, validloader,
                          checkpoint, output_dir, file_suffix)
