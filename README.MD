## AlexNet for Plankton classification ##

This is the code to train an AlexNet network modified to classify Plankton from a dataset of 3771 samples.
The training is done with batch SGDM over a cross-validation of 2 fold.

Directory structure:

- ./
-    assignment/
-        Initial code and assignment
-    dataset/
-        Datas_44.mat       Converted dataset to be compatible
-        Datas_44.mat.old   Original version of dataset, uncompatible with this python version
-    output/
-        models and accuracy saved to .pth and .txt files
-    runs/
-        file for tensorboard
-    skip/
-        preprocessed pattern saved to .pth files
-    src/
-        main.py
-        old_code.py


## MAIN.PY ##

## CONSTANT ##

- SKIP_PP     - skip the preprocessing (must be done at least one time to save pattern in ./skip/ folder)

- SKIP_TR     - skip the training (must be done at least one time)

- MODELS      - number of model to train, with 4 feature 4 models, note that lines 184-187 must be updated for more models

## PREPROCESSING ##

Each preprocessing function is defined from lines 39 to 116
each function is then called 184-187 and every pattern is then saved in ./skip/

to avoid the preprocessing to be done every time set SKIP_PP true.

## TRAINING ##

from line 266 to 324

if SKIP_TR is set to true load the available result saved in /outputs/ and skip training

at line 297 evalaute the results of the training and save the model

## ENSEMBLE ##

- Ensemble 1 345-366
- Ensemble 2 370-387
- Ensemble 3 390-412

## REQUIREMENTS ##
We used the code with:
- Python 3.11.5
- CUDA 12.1
- PyTorch 2.0.1

The following libraries are needed:
- scipy
- numpy
- tqdm
- torch
- torchvision
- skimage
