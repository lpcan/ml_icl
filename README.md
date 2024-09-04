# Measuring the intracluster light fraction with machine learning
This repository contains code and documentation that can be used to train, finetune, and run a machine learning model designed to predict the intracluster light (ICL) fraction in galaxy cluster images, as presented in Canepa et al. (in prep). 

The code has been tested on Python 3.9.6 and Python 3.9.17. The `requirements.txt` file describes the Python packages that are needed to run this code. 

## Model description
The basic structure of the model is comprised of an encoder which accepts the input images and encodes them into meaningful vector representations, and output layers which transform these into an ICL fraction prediction in the form of a probability distribution. The model architecture is described in `model.py`.

The encoder accepts 224x224 pixel images, and outputs a 256-dimensional vector representation of the image. This is a ResNet-50 model followed by a global average pooling layer. 

The output layers consist of one dense hidden layer with 2048 outputs, and an output layer with 48 outputs. These outputs are used as parameters to the output probability distribution, which is a mixture of beta distributions. The mode of the output distribution is taken as the most probable value of the ICL fraction.

The model was first trained on a dataset of artificially generated images, and then finetuned on a smaller dataset of real cluster images, measured using the surface brightness cut method.

## Description of the files
Below I briefly describe the structure of the repository and the files in it. You can safely skip this section unless you are interested in which files do what.
- `model.py`: this contains the `ImageRegressor` class, which defines the model architecture and is used to instantiate the model. This should not be edited unless you plan to fully retrain the model, as it will cause problems with loading from checkpoints.
- `augmentations.py`: this contains various custom augmentations that we apply to our training data, and defines the augmenters that are used by the model.
- `resnet_cifar10_v2.py`: this contains the definition for the ResNet-50 model our encoder is based on, originally from <a href="https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10_v2.py">here</a> with some modifications. This should not be edited unless you plan to fully retrain the model, as it will cause problems with loading from checkpoints.
- `prep_data.py`: this contains the code to prepare the various datasets for training, finetuning, and applying the model by resizing and rescaling the initial input datasets.
- `train.py`: this contains the code used to train the model on the initial training data.
- `finetune.py`: this contains the code used to finetune the model on the real cluster dataset. 
- `gradcam.py`: this contains code that is used to produce <a href="https://arxiv.org/abs/1610.02391">GradCAM</a> activation maps. The code is adapted from the implementation <a href="https://keras.io/examples/vision/grad_cam/">here</a>.
- `data/*`: this directory contains files related to downloading and preparing the data used in the model's training. Generally, you won't need to look at these files unless you are interested in how the data was downloaded (`create_combined_catalogue.py`, `download_cutouts.py`, `download_bulk.py`, `create_hdf.py`) or how the artificial ICL profiles were injected for the training data (`generate_train_data.py`). `display_cutouts.py` contains a way to flick through a set of cutouts in a very basic GUI.
- `measure_sb_cut/*`: this directory contains code related to doing measurements on the data with the surface brightness cut method. Generally, you won't need to look at these files unless you are interested in how the measurement is done. `measure.py` is used to automatically measure the training set. `measure_manual.py` is used to measure the real clusters. `measurement_helpers.py` contains functions that are common to both methods.

## Using this code
Below I describe how to set up and use this code in two possible use cases of the model, and the full method that can be used to reproduce the results from Canepa et al. (in prep). Model checkpoints and datasets for download can be found (TODO add Zenodo link)

### Applying the model directly to new data
TODO: Move these instructions into a notebook to make it clearer
1. Download the finetuned model checkpoint (`checkpoint-finetuned.zip`)
2. Prepare your data as a numpy array. You can use `prepare_data(path_to_images_file)`, which assumes that your data is in a HDF5 file organised by object ID, and each object contains two datasets: `'DATA'` (the data layer), and `'MASK'` (the mask layer) as downloaded from the HSC-SSP survey. If this does not apply to your data, you can prepare the data with the following steps:
    - resizing each image to a (224, 224, 1) array
    - stretch the image in a similar way to:
```python
    img = np.clip(img, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
```
- - apply the bright star mask to the image, such that the image is set to 0 where there is a mask
3. Create the model and load the weights using `model = load_model(model_name=checkpoint-finetuned)`. By default, this will look for the checkpoint file in the `checkpoints/` folder, but you can change this with the `path_prefix` argument, like `load_model(model_name=checkpoint-finetuned, path_prefix=path_to_checkpoint_directory)`. 

4. Run the model using `outputs = model(data)`, which will give you an array of Tensorflow Probability distributions. To find the mode of each distribution and thus the final predictions, run the below code:
```python
x = np.arange(0, 0.6, 0.0005)
logps = []
logcs = []
for i in x:
    logps.append(outputs.log_prob(i).numpy())
    logcs.append(outputs.log_cdf(i).numpy())
logps = np.stack(logps)
logcs = np.stack(logcs)
predictions = x[np.exp(logps).argmax(axis=0)]
```

### Finetuning the model on new data
TODO: add demo of these steps in a notebook
TODO

### Reproducing the full training from the paper
TODO

### Other edits
TODO

## Abstract
TODO
