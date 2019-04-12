import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.abspath('../'))


class MultiTaskNet(nn.Module):

    def __init__(self, input_size, input_channels, output_size, model_size):
        super(MultiTaskNet, self).__init__()

        self.input_conv_layer = nn.Sequential(
            nn.Conv1d(input_channels, 2 * model_size, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm1d(2 * model_size),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2)
        )

        self.hidden_conv_layer_1 = nn.Sequential(
            nn.Conv1d(2 * model_size, 2 * model_size, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(2 * model_size),
            nn.ReLU(True),
            nn.MaxPool1d(3, 3, 1)
        )

        self.hidden_conv_layer_2 = nn.Sequential(
            nn.Conv1d(2 * model_size, 2 * model_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(2 * model_size),
            nn.ReLU(True)
        )

        self.hidden_conv_layer_3 = nn.Sequential(
            nn.Conv1d(4 * model_size, 4 * model_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4 * model_size),
            nn.ReLU(True)
        )

        self.hidden_conv_layer_4 = nn.Sequential(
            nn.Conv1d(4 * model_size, 4 * model_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4 * model_size),
            nn.ReLU(True)
        )

        self.output_layer_input = int(((input_size/2)+2)/3) * 2 * model_size

        self.hidden_dense_layer_5 = nn.Sequential(
            nn.Linear(self.output_layer_input, 20 * model_size),
            nn.BatchNorm1d(20 * model_size),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.hidden_dense_layer_6 = nn.Sequential(
            nn.Linear(20 * model_size, 20 * model_size),
            nn.BatchNorm1d(20 * model_size),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(20 * model_size, output_size)
        )

        self.ID_output_layer = nn.Sequential(
            nn.Linear(20 * model_size, 10 * model_size),
            nn.BatchNorm1d(10 * model_size),
            nn.ReLU(True),

            nn.Linear(10 * model_size, 5 * model_size),
            nn.BatchNorm1d(5 * model_size),
            nn.ReLU(True),

            nn.Linear(5 * model_size, output_size - 3),
        )

        self.PR_output_layer = nn.Sequential(
            nn.Linear(20 * model_size, 10 * model_size),
            nn.BatchNorm1d(10 * model_size),
            nn.ReLU(True),

            nn.Linear(10 * model_size, 5 * model_size),
            nn.BatchNorm1d(5 * model_size),
            nn.ReLU(True),

            nn.Linear(5 * model_size, 1)
        )

        self.RT_output_layer = nn.Sequential(
            nn.Linear(20 * model_size, 10 * model_size),
            nn.BatchNorm1d(10 * model_size),
            nn.ReLU(True),

            nn.Linear(10 * model_size, 5 * model_size),
            nn.BatchNorm1d(5 * model_size),
            nn.ReLU(True),

            nn.Linear(5 * model_size, 1)
        )

        self.RR_output_layer = nn.Sequential(
            nn.Linear(20 * model_size, 10 * model_size),
            nn.BatchNorm1d(10 * model_size),
            nn.ReLU(True),

            nn.Linear(10 * model_size, 5 * model_size),
            nn.BatchNorm1d(5 * model_size),
            nn.ReLU(True),

            nn.Linear(5 * model_size, 1)
        )

    def forward(self, input):

        output = self.input_conv_layer(input)

        # conv layers
        output = self.hidden_conv_layer_1(output)
        output = self.hidden_conv_layer_2(output)

        # prepare output for dense layer
        output = output.view(output.shape[0], -1)

        # dense layers
        output = self.hidden_dense_layer_5(output)
        output = self.hidden_dense_layer_6(output)

        ID_output = self.ID_output_layer(output)
        PR_output = torch.exp(self.PR_output_layer(output))
        RT_output = torch.exp(self.RT_output_layer(output))
        RR_output = torch.exp(self.RR_output_layer(output))

        total_output = torch.cat((ID_output, PR_output, RT_output, RR_output), dim=1)

        return total_output
