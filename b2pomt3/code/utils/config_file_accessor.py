import configparser


class ConfigurationFileLoader:

    def __init__(self, config_file_path):
        self.config = configparser.ConfigParser()
        self.config.read("./" + config_file_path + ".in")

    def get_hyperparameters(self):
        # from a ConfigParser() file; get hyperparameters
        # return them in a dict
        hyperparam = {}
        hyperparam['use_gpu'] = \
            self.config.get('general', 'use_gpu') == 'True'
        hyperparam['generate_segemented_dataset'] = \
            self.config.get('general', 'generate_segemented_dataset') == 'True'
        hyperparam['train_ae'] = \
            self.config.get('general', 'train_autoencoder') == 'True'

        hyperparam['segmented_unlabeled_path'] = \
            self.config.get('path', 'segmented_unlabeled_dataset')
        hyperparam['segmented_labeled_path'] = \
            self.config.get('path', 'segmented_labeled_dataset')

        hyperparam['learning_rate'] = \
            float(self.config.get('optimizer', 'learning_rate'))
        hyperparam['momentum'] = \
            float(self.config.get('optimizer', 'momentum'))
        hyperparam['batchsize'] = \
            int(self.config.get('optimizer', 'batch_size'))
        hyperparam['nepoch'] = \
            int(self.config.get('optimizer', 'nepoch'))
        hyperparam['model'] = \
            self.config.get('model', 'name')
        hyperparam['hidden_size'] = \
            int(self.config.get('model', 'hidden_size'))
        hyperparam['dropout'] = \
            float(self.config.get('model', 'dropout'))
        hyperparam['n_layers'] = \
            int(self.config.get('model', 'n_layers'))
        hyperparam['kernel_size'] = \
            int(self.config.get('model', 'kernel_size'))
        hyperparam['pool_size'] = \
            int(self.config.get('model', 'pool_size'))
        hyperparam['tbpath'] = \
            self.config.get('path', 'tensorboard')
        hyperparam['modelpath'] = \
            self.config.get('path', 'model')
        hyperparam['traning_set_path'] = \
            self.config.get('path', 'training_dataset')
        hyperparam['validation_set_path'] = \
            self.config.get('path', 'validation_dataset')
        hyperparam['unlabeled_set_path'] = \
            self.config.get('path', 'unlabeled_dataset')
        hyperparam['dataloader_num_workers'] = \
            int(self.config.get('loader', 'num_workers'))
        weight1 = float(self.config.get('loss', 'weight1'))
        weight2 = float(self.config.get('loss', 'weight2'))
        weight3 = float(self.config.get('loss', 'weight3'))
        weight4 = float(self.config.get('loss', 'weight4'))
        hyperparam['weight'] = \
            [weight1, weight2, weight3, weight4]

        # For autoencoder
        hyperparam['ae_model'] = \
            self.config.get('autoencoder', 'model')
        hyperparam['ae_hidden_size'] = \
            list(map(int, self.config.get('autoencoder', 'hidden_size').split(',')))
        hyperparam['ae_kernel_size'] = \
            list(map(int, self.config.get('autoencoder', 'kernel_size').split(',')))
        hyperparam['ae_stride'] = \
            list(map(int, self.config.get('autoencoder', 'stride').split(',')))
        hyperparam['ae_padding'] = \
            list(map(int, self.config.get('autoencoder', 'padding').split(',')))
        hyperparam['ae_batch_size'] = \
            int(self.config.get('autoencoder', 'batch_size'))
        hyperparam['ae_n_epochs'] = \
            int(self.config.get('autoencoder', 'n_epochs'))
        hyperparam['ae_lr'] = \
            float(self.config.get('autoencoder', 'learning_rate'))
        hyperparam['ae_criterion'] = \
            self.config.get('autoencoder', 'criterion')
        hyperparam['ae_optimizer'] = \
            self.config.get('autoencoder', 'optimizer')
        hyperparam['ae_checkpoint'] = \
            self.config.get('autoencoder', 'checkpoint')
        hyperparam['ae_output_dir'] = \
            self.config.get('autoencoder', 'output_dir')
        hyperparam['ae_file_suffix'] = \
            self.config.get('autoencoder', 'file_suffix')

        hyperparam['encoder_transform'] = \
            self.config.get('encoder', 'encoder_transform') == 'True'
        hyperparam['encoder_path'] = \
            self.config.get('encoder', 'encoder_path')
        hyperparam['encoder_model'] = \
            self.config.get('encoder', 'encoder_model')

        return hyperparam
