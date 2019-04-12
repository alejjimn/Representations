import math
import torch.nn as nn


class AutoEncoder_2l(nn.Module):
    """ Class for 2 layers AutoEncoder """
    def __init__(self,
                 input_size=1,
                 hidden_size=[32, 36],
                 kernel_size=[5, 5, 5, 5],
                 padding=[1, 2, 2, 1],
                 stride=[1, 1, 1, 1],
                 pool_size=2):
        """
        Args:
            input_size (int): size of initial sample
            hidden_size (list): list of number of kernels for each layers
            kernel_size (list): list of kernel size for each layers
            padding (list): list of padding size for each layers
            stride (list): list of stride size for each layers
            pool_size (int): list of pooling size for each layers

        """
        super(AutoEncoder_2l, self).__init__()
        # encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size[0], kernel_size[0], padding=padding[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool1d(pool_size, return_indices=True)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_size[0], hidden_size[1], kernel_size[1], padding=padding[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool1d(pool_size, return_indices=True)

        # Compute output size
        encoder1_size = int(math.ceil(math.ceil(((230 + 2*padding[0] - kernel_size[0]) / stride[0]) + 1) / 2))
        encoder2_size = int(math.ceil(math.ceil(((encoder1_size + 2*padding[1] - kernel_size[1]) / stride[1]) + 1) / 2))
        flatten_size = encoder2_size * hidden_size[1]
        fc1_size = int(math.floor(flatten_size / 2))
        emb_size = int(math.floor(fc1_size / 2))

        self.fc1 = nn.Linear(flatten_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, emb_size)
        self.fc3 = nn.Linear(emb_size, fc1_size)
        self.fc4 = nn.Linear(fc1_size, flatten_size)

        # decoder
        self.unpool1 = nn.MaxUnpool1d(pool_size)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[1], hidden_size[0], kernel_size[2], padding=padding[2]),
            nn.ReLU(True)
        )
        self.unpool2 = nn.MaxUnpool1d(pool_size)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[0], input_size, kernel_size[3], padding=padding[3]),
            nn.ReLU(True)
        )

    def forward(self, x):
        """
         Forward pass of AutoEncoder_2l

            Args:
                x (torch.tensor): data
            Returns:
                encoded (torch.tensor): encoded data
                decoded (torch.tensor): decoded data
        """
        # encoder
        x = self.encoder1(x)
        x, indices1 = self.pool1(x)
        x = self.encoder2(x)
        x, indices2 = self.pool2(x)

        # get latent representation
        h = x.size(1)
        o = x.size(2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        encoded = self.fc2(x)

        # decoder
        x = self.fc3(encoded)
        x = self.fc4(x)
        x = x.view(x.size(0), h, o)
        x = self.unpool1(x, indices2)
        x = self.decoder1(x)
        x = self.unpool2(x, indices1)
        decoded = self.decoder2(x)
        return encoded, decoded


class AutoEncoder_3l(nn.Module):
    """ Class for 3 layers AutoEncoder with 3 layers of pooling """
    def __init__(self,
                 input_size=1,
                 hidden_size=[32, 48, 64],
                 kernel_size=[5, 5, 5, 5, 5],
                 padding=[1, 2, 2, 2, 1],
                 stride=[1, 1, 1, 1, 1],
                 pool_size=2):
        super(AutoEncoder_3l, self).__init__()
        """
        Args:
            input_size (int): size of initial sample
            hidden_size (list): list of number of kernels for each layers
            kernel_size (list): list of kernel size for each layers
            padding (list): list of padding size for each layers
            stride (list): list of stride size for each layers
            pool_size (int): list of pooling size for each layers

        """
        # encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size[0], kernel_size[0],
                      padding=padding[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool1d(pool_size, return_indices=True)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_size[0], hidden_size[1], kernel_size[1],
                      padding=padding[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool1d(pool_size, return_indices=True)

        self.encoder3 = nn.Sequential(
            nn.Conv1d(hidden_size[1], hidden_size[2], kernel_size[2],
                      padding=padding[2]),
            nn.BatchNorm1d(hidden_size[2]),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool1d(pool_size, return_indices=True)

        # Compute output size
        encoder1_size = int(math.floor(math.floor(((230 + 2*padding[0] - kernel_size[0]) / stride[0]) + 1) / 2))
        encoder2_size = int(math.floor(math.floor(((encoder1_size + 2*padding[1] - kernel_size[1]) / stride[1]) + 1) / 2))
        encoder3_size = int(math.floor(math.floor(((encoder2_size + 2*padding[2] - kernel_size[2]) / stride[2]) + 1) / 2))
        flatten_size = encoder3_size * hidden_size[2]
        fc1_size = int(math.floor(flatten_size / 2))
        emb_size = int(math.floor(fc1_size / 2))

        # fully connected layers
        self.fc1 = nn.Linear(flatten_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, emb_size)
        self.fc3 = nn.Linear(emb_size, fc1_size)
        self.fc4 = nn.Linear(fc1_size, flatten_size)
        # decoder
        self.unpool1 = nn.MaxUnpool1d(pool_size)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[2], hidden_size[1], kernel_size[3],
                               padding=padding[3]),
            nn.ReLU(True)
        )
        self.unpool2 = nn.MaxUnpool1d(pool_size)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[1], hidden_size[0], kernel_size[4],
                               padding=padding[4]),
            nn.ReLU(True)
        )
        self.unpool3 = nn.MaxUnpool1d(pool_size)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[0], input_size, kernel_size[5],
                               padding=padding[5]),
            nn.ReLU(True)
        )
        self.out = nn.Tanh()

    def forward(self, x):
        """
         Forward pass of AutoEncoder_3l

            Args:
                x (torch.tensor): data
            Returns:
                encoded (torch.tensor): encoded data
                decoded (torch.tensor): decoded data
        """
        # encoder
        x = self.encoder1(x)
        x, indices1 = self.pool1(x)
        x = self.encoder2(x)
        x, indices2 = self.pool2(x)
        x = self.encoder3(x)
        x, indices3 = self.pool2(x)

        # get latent representation
        h = x.size(1)
        o = x.size(2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        encoded = self.fc2(x)

        # decoder
        x = self.fc3(encoded)
        x = self.fc4(x)
        x = x.view(x.size(0), h, o)
        x = self.unpool1(x, indices3)
        x = self.decoder1(x)
        x = self.unpool2(x, indices2)
        x = self.decoder2(x)
        x = self.unpool3(x, indices1)
        decoded = self.decoder3(x)
        return encoded, decoded


class AutoEncoder_3l_v2(nn.Module):
    """ Class for 3 layers AutoEncoder with 2 layers of pooling """
    def __init__(self,
                 input_size=1,
                 hidden_size=[32, 48, 64],
                 kernel_size=[13, 9, 5, 5, 10, 13],
                 padding=[6, 4, 2, 2, 4, 6],
                 stride=[1, 1, 1, 1, 1, 1],
                 pool_size=2):
        super(AutoEncoder_3l_v2, self).__init__()
        """
            Args:
                input_size (int): size of initial sample
                hidden_size (list): list of number of kernels for each layers
                kernel_size (list): list of kernel size for each layers
                padding (list): list of padding size for each layers
                stride (list): list of stride size for each layers
                pool_size (int): list of pooling size for each layers

        """

        # encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size[0], kernel_size[0],
                      padding=padding[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool1d(pool_size, return_indices=True)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_size[0], hidden_size[1], kernel_size[1],
                      padding=padding[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool1d(pool_size, return_indices=True)

        self.encoder3 = nn.Sequential(
            nn.Conv1d(hidden_size[1], hidden_size[2], kernel_size[2], padding=padding[2]),
            nn.BatchNorm1d(hidden_size[2]),
            nn.ReLU(True)
        )

        # Compute output size
        encoder1_size = int(math.floor(math.floor(((230 + 2*padding[0] - kernel_size[0]) / stride[0]) + 1) / 2))
        encoder2_size = int(math.floor(math.floor(((encoder1_size + 2*padding[1] - kernel_size[1]) / stride[1]) + 1) / 2))
        encoder3_size = math.floor(((encoder2_size + 2*padding[2] - kernel_size[2]) / stride[2]) + 1)
        flatten_size = encoder3_size * hidden_size[2]
        fc1_size = int(math.floor(flatten_size / 2))
        emb_size = int(math.floor(fc1_size / 2))

        # fully connected layers
        self.fc1 = nn.Linear(flatten_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, emb_size)
        self.fc3 = nn.Linear(emb_size, fc1_size)
        self.fc4 = nn.Linear(fc1_size, flatten_size)

        # decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[2], hidden_size[1], kernel_size[3], padding=padding[3]),
            nn.ReLU(True)
        )
        self.unpool1 = nn.MaxUnpool1d(pool_size)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[1], hidden_size[0], kernel_size[4], padding=padding[4]),
            nn.ReLU(True)
        )
        self.unpool2 = nn.MaxUnpool1d(pool_size)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size[0], input_size, kernel_size[5], padding=padding[5]),
            nn.ReLU(True)
        )

    def forward(self, x):
        """
         Forward pass of AutoEncoder_3l_2w

            Args:
                x (torch.tensor): data
            Returns:
                encoded (torch.tensor): encoded data
                decoded (torch.tensor): decoded data

        """
        # encoder
        x = self.encoder1(x)
        x, indices1 = self.pool1(x)
        x = self.encoder2(x)
        x, indices2 = self.pool2(x)
        x = self.encoder3(x)

        # get latent representation
        h = x.size(1)
        o = x.size(2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        encoded = self.fc2(x)

        # decoder
        x = self.fc3(encoded)
        x = self.fc4(x)
        x = x.view(x.size(0), h, o)
        x = self.decoder1(x)
        x = self.unpool1(x, indices2)
        x = self.decoder2(x)
        x = self.unpool2(x, indices1)
        decoded = self.decoder3(x)
        return encoded, decoded


def weights_init(model):
    """
     Initialize weights and biases.

        Args:
            model (nn.Module): model to have weights initialized
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
        model.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
