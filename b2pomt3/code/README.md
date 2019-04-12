# IFT6759 Winter 2019
## OM Signal Project Block 2

Authors:
Éric Girard
Yasmine Dumouchel
Benjamin Rosa

Project Organization
------------

    ├── checkpoint        <- models saved during training
    ├── dataset           <- custom dataset and transforms
    ├── models            <- different prediction modules used for the 4 tasks
    ├── notebooks         <- notebooks used mainly for quick testing
    ├── utils             <- utilities used accross the application
    └── README.md         <- The top-level README for developers using this project.

--------

## Usage:

train_autoencoder.py : trains an autoencoder model

python train_autoencoder.py --config input.in

--config is the path to the configuration file

## Task assignation conventions
* Go in the Projects tab from this repo Github  
* Select OMSignal  
* Move a task from the 'To do' column to the 'In progress' column
* Convert the task into an issue, and assign the issue to you
* Once the task is done, do the pull request (see Github conventions)
* Move the task from the 'In progress' column to 'Testing/Code review'
* Unassign the issue from you
* The person doing the code review will assign the task to herself
* Once the code review is done, the code is merged into master
* The person who made the code review move the task from 'Testing' to 'Done'

## Github conventions
* Each feature must have his own branch for development
  * git checkout -b nameOfTheNewBranch
* When changes are made, push is only made to the feature's branch
  * git add .
  * git commit -m "Relevant message regarding the changes"
  * git checkout master
  * git pull --rebase
  * git checkout nameOfTheNewBranch
  * git merge master
  * Fix conflicts if there is any
  * git push origin nameOfTheNewBranch
* Once the changes in a personal branch are ready, do a pull request for the master branch
  * go to the github page of the project https://github.com/BenjaminBenoit/OMSignal_Block2
  * select your branch using the drop down button
  * click on create pull request
  * put master branch as the head
  * confirm the creation of the pull request

## Writing Conventions
* Private class methods name should start with an underscore (Pep8 convention)
* Limit all lines to a maximum of 79 characters. Only exception is when it impact negatively the readibility of a sentence.
* For flowing long blocks of text with fewer structural restrictions (docstrings or comments), the line length should be limited to 72 characters.

## Configuration file

### Configuration file guide
The default configuration file is ./input.in
This default configuration file should not be modify by the users.
It is only here to provide a template of what the program is expected to have.
Everyone should have a config file outside the project with his own parameters.
When launching the program, add the --config arguments in the command line to specify the path to your config file.

If someone change the config file, it is important to update input.in accordingly.
So that everyone would be able to do the changes in their own config file.

### Configuration file parameters
The configuration file is an input file containing several parameters :
* general.use_gpu             : boolean - whether or not we use the GPU for training/validation/testing
* general.generate_segemented_dataset : boolean - whether or not we generate a new segmented dataset before starting the training
                                if True, it is this new segmented dataset who will be used during the training.
* general.train_autoencoder   : boolean - whether or not we train a new auto-encoder and use it before the prediction module.
* optimizer.learning_rate     : float - learning rate for the optimizer
* optimizer.momentum          : float - momentum for the optimizer
* optimizer.nepoch            : int - number of epochs
* optimizer.batch_size        : int - batch size
* loader.num_workers          : int - number of workers in dataloader
* model.name                  : string - name of the model we want to use for the prediction module
* model.hidden_size           : int - size of the hidden layer(s) for the prediction module
* model.dropout               : float - prediction module dropout value
* model.n_layers              : int - number of layers for the prediction module
* model.kernel_size           : int - prediction module kernel size
* model.pool_size             : int - prediction module pooling size
* loss.weight1                : int - loss factor for task 1 in multitask prediction module
* loss.weight2                : int - loss factor for task 2 in multitask prediction module
* loss.weight3                : int - loss factor for task 3 in multitask prediction module
* loss.weight4                : int - loss factor for task 4 in multitask prediction module
* autoencoder.model           : string - name of the model we want to use as autoencoder
* autoencoder.hidden_size     : string - size of each hidden layers of the autoencoder. separated by a ','
* autoencoder.kernel_size     : string - size of each kernel for each layer of the autoencoder. separated by a ','
* autoencoder.stride          : string - size for the stride for each layer of the autoencoder. separated by a ','
* autoencoder.padding         : string - padding size for each layer of the autoencoder. separated by a ','
* autoencoder.batch_size      : int - batch size 
* autoencoder.n_epochs        : int - number of epochs
* autoencoder.criterion       : string - loss criterion
* autoencoder.optimizer       : string - weight initialization method
* autoencoder.learning_rate   : float - learning rate for the autoencoder optimizer
* autoencoder.output_dir      : string - path for the checkpoints
