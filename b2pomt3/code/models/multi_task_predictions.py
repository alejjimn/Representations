import sys
import torch
import numpy as np
from scipy.stats import mode
sys.path.append('../')
from dataset.to_float_tensor_transform import ToFloatTensorTransform


def multi_task_predict(model, dataset, encoder=None):
    """
    make label predictions with multitask model on unsegmented dataset
    ----------
     model : MultiTaskNet
            model we want to output the prediction for
    dataset : torch.util.data.Datasets
       UNSEGMENTED (3750 or 3754 dimensions) dataset object (OMSignalDataset)
    """
    num_participants = 32

    # list of predicted labels for test dataset
    predicted_labels = []
    y_true = []

    to_float = ToFloatTensorTransform()

    test_data = dataset
    # No need to compute gradients for validation
    with torch.no_grad():
        # Activate eval mode for the model
        model.eval()

        # Predict for all segments of sample
        for sample, target in test_data:
            batch = None
            for segmented_sample in sample:
                segmented_sample = to_float(segmented_sample)
                if encoder is not None:

                    segmented_sample = encoder(segmented_sample)

                if batch is None:
                    batch = segmented_sample
                else:
                    batch = torch.cat((batch,  segmented_sample), dim=0)

            input = torch.unsqueeze(batch, 1)

            # Forward pass in the model
            output = model(input)

            # Columns 1 to 32 are ID prediction, 33 is PR, 34 is RT and 35 is RR
            ID_predictions = output[:, :num_participants]
            ID_predictions = torch.max(ID_predictions, 1)[1].cpu().numpy()
            PR_predictions = output[:, num_participants].squeeze().cpu().detach().numpy()
            RT_predictions = output[:, num_participants + 1].squeeze().cpu().detach().numpy()
            RR_predictions = output[:, num_participants + 2].squeeze().cpu().detach().numpy()

            # test average segement predictions
            ID_mode = mode(ID_predictions)[0][0]
            pred_labels = np.array([PR_predictions.mean(),
                                    RT_predictions.mean(),
                                    RR_predictions.max(),
                                    ID_mode]).T

            predicted_labels.append(pred_labels)
            y_true.append(target)

    # convert list of predicted labels into numpy array
    y_pred = np.vstack(predicted_labels).reshape(
        (-1, 4)).astype(np.float32)

    y_true = np.vstack(y_true).reshape(
        (-1, 4)).astype(np.float32)

    return y_pred, y_true
