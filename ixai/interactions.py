import torch
import torch.nn as nn


def explanation_loss(model, data, attributor, correct_explanations, loss_function=None):
    """
    Compute the loss between the attributions of the model and the correct explanations
    :param model: The model to optimize.
    :param data:  [X, y] where X is the input data and y is the target
    :param attributor: A function that takes the model, X and y and returns the attributions of the model
    :param correct_explanations: The correct explanations for each input in data
    :param loss_function: Which loss function to use on the explanations. Default is nn.MSELoss
    :return: A pytorch differentiable loss on the explanations
    """
    assert torch.is_tensor(data[0]), "Only implemented for pytorch"
    assert torch.is_tensor(data[1]), "Only implemented for pytorch"

    correct_explanations = correct_explanations.detach()  # make sure we do not compute gradients for this

    X, y = data

    loss_function = nn.MSELoss() if loss_function is None else loss_function

    model_state = model.training

    def reset_model():
        model.train() if model_state else model.eval()

    model.eval()  # put model in eval mode to compute attributions

    atts = attributor(model, X, y)

    atts = atts.reshape(correct_explanations.size())



    reset_model()
    return loss_function(atts, correct_explanations)
