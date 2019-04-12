import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))


class MultiTaskTrainer():

    def __init__(self, model, train_loader, valid_loader, save_path,
                 learning_rate=0.0001,
                 batch_size=16,
                 weight_decay=0,
                 log=True, seed=111):
        """
        Trainer for the multi_task_model
        Parameters
        ----------
         model : MultiTaskNet
                model we want to do the training for
        train_loader : torch.util.data.Datasets
           dataset with the training data
        valid_loader : torch.util.data.Datasets
           dataset with the validation data
        save_path : String
            path to the folder where we want to save the results of the training
        learning_rate : float
            learning rate to use during the training
        batch_size : int
            batch size to use during the training
        weight_decay : int
            weight decay to use during the training
        log : boolean
            whether or not we log information during the training
        seed : int
            seed used for shuffling the data
        """

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model.to(self.device)

        self.save_path = save_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_participants = 32

        # Create directory for the results if it doesn't already exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Delete old log file and start logging
        if log:
            log_file = "train.log"
            if os.path.exists(log_file):
                os.remove(log_file)
            logging.basicConfig(filename=log_file, level=logging.INFO)
            logging.getLogger().addHandler(logging.StreamHandler())

        self.mean_epoch_train_losses = {
            "ID": [],
            "PR": [],
            "RT": [],
            "RR": [],
            "total": []
        }
        self.mean_epoch_valid_losses = {
            "ID": [],
            "PR": [],
            "RT": [],
            "RR": [],
            "total": []
        }
        self.accuracies = []

        self.ID_loss = nn.CrossEntropyLoss()
        self.PR_loss = nn.L1Loss()  # nn.MSELoss()
        self.RT_loss = nn.L1Loss()
        self.RR_loss = nn.L1Loss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

    def train(self, nb_epochs=10):
        logging.info("Start training.")

        # Activate train mode for the model
        self.model.train()

        # Training loop over each epoch
        for epoch in range(nb_epochs):
            logging.info("Epoch : " + str(epoch))

            losses = {
                "ID": [],
                "PR": [],
                "RT": [],
                "RR": [],
                "total": []
            }

            for batch_id, (input, label) in enumerate(self.train_loader):
                #print("Batch: ",batch_id)
                #print("Input: ",input.size())
                #print("Label: ",label.size())
                # Get the labels and put on detected device (cpu or gpu)
                PR_mean_truth = label[:, 0].to(self.device, dtype=torch.float)
                RT_mean_truth = label[:, 1].to(self.device, dtype=torch.float)
                RR_std_deviation_truth = label[:, 2].to(self.device, dtype=torch.float)
                participant_id_label = label[:, 3].to(self.device, dtype=torch.float)

                input = input.to(self.device, dtype=torch.float)

                # Reset the gradients
                self.model.zero_grad()

                # Forward pass in the model
                output = self.model.forward(input)

                # Columns 1 to 32 are ID prediction, 33 is PR, 34 is RT and 35 is RR
                ID_prediction = output[:, :self.num_participants]
                PR_prediction = output[:, self.num_participants]
                RT_prediction = output[:, self.num_participants+1]
                RR_prediction = output[:, self.num_participants+2]

                # Evaluate losses for the classification task, and the three regressions tasks
                ID_loss = self.ID_loss(ID_prediction, participant_id_label.long())
                PR_loss = self.PR_loss(PR_prediction, PR_mean_truth)
                RT_loss = self.RT_loss(RT_prediction, RT_mean_truth)
                RR_loss = self.RR_loss(RR_prediction, RR_std_deviation_truth)

                # Add all losses and backpropagate
                total_loss = 0.8 * ID_loss + 0.2 * (PR_loss + RT_loss + RR_loss)
                total_loss = total_loss
                total_loss.backward()
                self.optimiser.step()

                # Keep track of the loss for each batch of this epoch
                losses["total"].append(total_loss.item())
                losses["ID"].append(ID_loss.item())
                losses["PR"].append(PR_loss.item())
                losses["RT"].append(RT_loss.item())
                losses["RR"].append(RR_loss.item())

            # Keep track of the mean of the losses for each epoch
            self.mean_epoch_train_losses["total"].append(torch.mean(torch.tensor(losses["total"])).item())
            self.mean_epoch_train_losses["ID"].append(torch.mean(torch.tensor(losses["ID"])).item())
            self.mean_epoch_train_losses["PR"].append(torch.mean(torch.tensor(losses["PR"])).item())
            self.mean_epoch_train_losses["RT"].append(torch.mean(torch.tensor(losses["RT"])).item())
            self.mean_epoch_train_losses["RR"].append(torch.mean(torch.tensor(losses["RR"])).item())

            # Check accuracy on validation set
            self.validate()

            # Save model if it has the smallest total loss to date
            if len(self.mean_epoch_train_losses["total"]) > 1 and self.mean_epoch_train_losses["total"][-1] < torch.min(torch.tensor(self.mean_epoch_train_losses["total"][:-1])):
                self.save_model({
                   "multi_task_net": self.model
                })

            # Output epoch loss information
            logging.info("Epoch train total loss : " + str(self.mean_epoch_train_losses["total"][-1]))
            logging.info("Epoch valid total loss : " + str(self.mean_epoch_valid_losses["total"][-1]))

            logging.info("Epoch train ID loss : " + str(self.mean_epoch_train_losses["ID"][-1]))
            logging.info("Epoch valid ID loss : " + str(self.mean_epoch_valid_losses["ID"][-1]))

            logging.info("Epoch train PR loss : " + str(self.mean_epoch_train_losses["PR"][-1]))
            logging.info("Epoch valid PR loss : " + str(self.mean_epoch_valid_losses["PR"][-1]))

            logging.info("Epoch train RT loss : " + str(self.mean_epoch_train_losses["RT"][-1]))
            logging.info("Epoch valid RT loss : " + str(self.mean_epoch_valid_losses["RT"][-1]))

            logging.info("Epoch train RR loss : " + str(self.mean_epoch_train_losses["RR"][-1]))
            logging.info("Epoch valid RR loss : " + str(self.mean_epoch_valid_losses["RR"][-1]))

            # Keep track of the accuracy
            valid_accuracy = self.get_ID_accuracy(self.valid_loader)
            self.accuracies.append(valid_accuracy)

            logging.info("Accuracy on train data : " + str(self.get_ID_accuracy(self.train_loader)) + " %")
            logging.info("Accuracy on valid data : " + str(valid_accuracy) + " %")

            # Plot accuracy and loss
            self.save_results(self.mean_epoch_train_losses, self.mean_epoch_valid_losses)

        # Plot accuracy and loss
        best_accuracy = max(self.accuracies)
        best_loss = min(self.mean_epoch_valid_losses["total"])
        return best_accuracy, best_loss

    def validate(self):
        logging.info("Start validation.")

        valid_losses = {
            "ID": [],
            "PR": [],
            "RT": [],
            "RR": [],
            "total": []
        }

        # No need to compute gradients for validation
        with torch.no_grad():
            # Activate eval mode for the model
            self.model.eval()

            # Only one big batch for validation
            for batch_id, (input, label) in enumerate(self.valid_loader):

                # Get the labels and put on detected device (cpu or gpu)
                PR_mean_truth = label[:, 0].to(self.device, dtype=torch.float)
                RT_mean_truth = label[:, 1].to(self.device, dtype=torch.float)
                RR_std_deviation_truth = label[:, 2].to(self.device, dtype=torch.float)
                participant_id_label = label[:, 3].to(self.device, dtype=torch.float)
                input = input.to(self.device, dtype=torch.float)

                # Forward pass in the model
                output = self.model(input)

                # Columns 1 to 32 are ID prediction, 33 is PR, 34 is RT and 35 is RR
                ID_prediction = output[:, :self.num_participants]
                PR_prediction = output[:, self.num_participants]
                RT_prediction = output[:, self.num_participants+1]
                RR_prediction = output[:, self.num_participants+2]

                # Evaluate losses for the classification task, and the three regressions tasks
                ID_loss = self.ID_loss(ID_prediction, participant_id_label.long())
                PR_loss = self.PR_loss(PR_prediction, PR_mean_truth)
                RT_loss = self.RT_loss(RT_prediction, RT_mean_truth)
                RR_loss = self.RR_loss(RR_prediction, RR_std_deviation_truth)

                current_total_loss = 0.8 * ID_loss + 0.2 * (PR_loss + RT_loss + RR_loss)

                # Keep track of the loss for each batch of this epoch
                valid_losses["total"].append(current_total_loss.item())
                valid_losses["ID"].append(ID_loss.item())
                valid_losses["PR"].append(PR_loss.item())
                valid_losses["RT"].append(RT_loss.item())
                valid_losses["RR"].append(RR_loss.item())

            # Keep track of the loss
            self.mean_epoch_valid_losses["ID"].append(torch.mean(torch.tensor(valid_losses["ID"])).item())
            self.mean_epoch_valid_losses["PR"].append(torch.mean(torch.tensor(valid_losses["PR"])).item())
            self.mean_epoch_valid_losses["RT"].append(torch.mean(torch.tensor(valid_losses["RT"])).item())
            self.mean_epoch_valid_losses["RR"].append(torch.mean(torch.tensor(valid_losses["RR"])).item())
            self.mean_epoch_valid_losses["total"].append(torch.mean(torch.tensor(valid_losses["total"])).item())

    def get_ID_accuracy(self, data_loader):
        # Activate eval mode for the model
        self.model.eval()

        total = 0
        correct = 0

        # No gradients computation (faster speed)
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(data_loader):

                # Get the labels and put on detected device (cpu or gpu)
                participant_id_label = label[:, 3].to(self.device, dtype=torch.float)
                input = input.to(self.device, dtype=torch.float)

                # Get the models' predictions
                output = self.model(input)
                ID_prediction = output[:, :self.num_participants]

                # Count the number of correct ID predictions
                ID_prediction = torch.max(ID_prediction.data, 1)[1]
                correct += (ID_prediction == participant_id_label.long()).sum().item()
                total += participant_id_label.shape[0]

        # Workaround to get a float result with python 2 from an int division (without importing division from future (python 3))
        accuracy = (correct / (total * 1.0)) * 100
        return accuracy

    def save_results(self, Ys_train_to_plot, Ys_valid_to_plot):

        for name in Ys_train_to_plot.keys():

            plt.xlabel("Epoch")
            plt.ylabel(name + " loss")
            plt.plot(Ys_train_to_plot[name], label="Train")
            plt.plot(Ys_valid_to_plot[name], label="Valid")
            plt.legend(loc="best")
            plt.savefig(self.save_path + name + ".png")
            plt.clf()

    def save_model(self, models):

        for name, model in models.items():
            logging.info("Saving model " + name + " to : " + str(self.save_path))
            torch.save(model.state_dict(), self.save_path + name + ".pt")

    def load_model(self, models):
        logging.info("Loading models from : " + str(self.save_path))

        for name, model in models.items():
            model.load_state_dict(torch.load(self.save_path + name + ".pt"))
