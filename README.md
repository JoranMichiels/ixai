# Interactive eXplainable Artificial Intelligence (IXAI)

## Install instructions

- clone the Github respository
- install Conda environment (requires Conda): `conda env create -f environment.yml`
- if you want to run the example, download the data files from
  this [drive](https://kuleuven-my.sharepoint.com/:f:/g/personal/joran_michiels_kuleuven_be/Etvn8hgDPCdJr4gi4Lf5Py4Bc4lVImxr_73bzY6z5hVBSQ?e=TqRZDB)
  and place them in the `clickmette` folder. Combine `10images.pt.aa` and `10images.pt.ab` back into one file:
  `cat 10images.pt.?? > 10images.pt`

## Files and important functions

- `ixai/torch_explainers.py` contains differentiable explainer classes, modified from [Captum](https://captum.ai/)
  explainer classes. The available explainers are:
    - class `SaliencyGraph`
    - class `IntegratedGradientsGraph`
    - class `GradientShapGraph` (also known as Expected Gradients)
    - class `DeepLiftGraph`
    - class `DeepLiftShapGraph` (also known as SHAP)
- `ixai/interactions.py`:
    - function `explanation_loss(...)` returns the pytorch loss for the model's explanation
- `ixai/models.py`:
    - class `ImageNette(...)` is a pytorch like model class that can be used to train a model on
      the [Imagenette](https://github.com/fastai/imagenette) dataset
      with [ClickMe](https://serre-lab.clps.brown.edu/resource/clickme/) explanations
- `ixai/data.py`:
    - class `ClickMetteDataset(...)` is a pytorch like dataset class that can be used to load the subset of
      the  [Imagenette](https://github.com/fastai/imagenette) dataset that has a
      [ClickMe](https://serre-lab.clps.brown.edu/resource/clickme) explanation. This will be the main training dataset
      for our example.
    - class `ImageNetteDataset(...)` is a pytorch like dataset class that can be used to load the
      complete [Imagenette](https://github.com/fastai/imagenette). We will use this in our example for extra validation.
- `helpers.py` contains some helper functions

## General Usage

- pick an explainer of your liking from `ixai/torch_explainers.py`
- implement a specific `attributor(model, input, target)` function that returns the final (normalized) explanation e.g.:
  ```python
  from ixai.torch_explainers import SaliencyGraph
  def saliency_attributor(model, input, target):
      explainer = SaliencyGraph(model)
      return explainer.attribute(input, target)
  ```
- use this attributor function to define a specific `exp_loss` based on `explanation_loss` in `ixai/interactions.py`
  e.g.:
    ```python
    from ixai.interactions import explanation_loss
    def exp_loss(model, data, exps):
        return explanation_loss(model, data, saliency_attributor, exps)
    ```
- train your model with this `exp_loss` as an additional loss term

## Example

An example is included in `example.py`. This script will train a ResNet on a subset of
the [Imagenette](https://github.com/fastai/imagenette)
dataset where each sample is accompanied by a correct explanation retrieved from
the [ClickMe](https://serre-lab.clps.brown.edu/resource/clickme/) dataset. For your convenience, the save files of this
dataset are available in
this [drive](https://kuleuven-my.sharepoint.com/:f:/g/personal/joran_michiels_kuleuven_be/Etvn8hgDPCdJr4gi4Lf5Py4Bc4lVImxr_73bzY6z5hVBSQ?e=TqRZDB).

## Troubleshooting

In case of problems or questions, please
contact [joran.michiels@esat.kuleuven.be](mailto:joran.michiels@esat.kuleuven.be) or submit an issue on the GitHub
repository.

## Further work

This repository is a work in progress. Future work will include:

- visualizations
- more examples
- more explainers
- a simple GUI for convenient correction of explanations
