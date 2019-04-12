import pickle


def save_model(model, model_path):
    """
    Save the model to a path as a pickle object
    :param model: The instance of the model to save
    :param model_path: The path where to save the model
    """
    with open(model_path, mode='wb') as file:
        pickle.dump(model, file)


def load_model(model_path):
    """
    Load a saved model
    :param model_path: The path of the model to load
    :return: The loaded model as a object
    """
    with open(model_path, mode='rb') as file:
        return pickle.load(file)
