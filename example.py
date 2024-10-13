import torch
from torch import nn

from helpers import normalize_tensor
from ixai.interactions import explanation_loss
from ixai.models import ImageNette
from ixai.torch_explainers import SaliencyGraph

# Make sure GPU is available
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Chose params
weight = 1


# Define the specific attributor
def saliency_attributor(model, in_X, in_y):
    explainer = SaliencyGraph(model)
    atts = explainer.attribute(in_X, target=in_y, abs=True)

    # average over color channels
    atts = torch.mean(atts.squeeze(), dim=1)

    # normalize attributions
    atts = normalize_tensor(atts, method='minmax', samplewise=True)

    return atts


# Define the specific explanation loss
def exp_loss(model, data, exps):
    return explanation_loss(model, data, saliency_attributor, exps, loss_function=nn.MSELoss())


net = ImageNette(debug=False)
net.cuda()

net.fit(exp_loss, n_epochs=5, weight=weight)
