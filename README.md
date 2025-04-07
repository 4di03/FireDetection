# FireDetection

## Group Members
Adithya Palle

## Project Description

Detecting fires rapidly in both indoor and outdoor
environments is essential for minimizing damage and ensuring
timely response. We implement two vision-based models that predict
the presence of fire in each frame of a real-time video feed.
The first is a lightweight convolutional neural network (CNN)
inspired by AlexNet. The second is an optical flow-
based model which calculates the percentage of flow in flame-
colored regions. We evaluate both models on a test set of videos 
and determine the recall, false positive rate, and performance (in frames per second).
We aim to create a lightweight model that can run on CPU-only devices
and achieve a high recall in classifiying images as fire or no-fire.

## Data

To run the notebooks, you will need the appropriate data in your data folder. 

Below are the sources for the datasets required in `data_extraction.py`.

The Places365 needed for `PLACES_DATA_PATH`, you can find it [here](https://github.com/CSAILVision/places365).

The Fire videos validaiton and test data for `FIRE_VIDEOS_DATA_PATH` can be downloaded [here](https://drive.google.com/file/d/1-Q5WJyw4Lil0-Ww_tmM23NFRcRrT2kbz/view?usp=sharing). The drive folder is just a split version of the [FIRESENSE](https://www.kaggle.com/datasets/chrisfilo/firesense) dataset.

The Fire image dataset for `FIRE_IMAGE_DATA_PATH` can be found [here](https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/dataset/1).


`TENSOR_CACHE_PATH` should be initialized empty and will be filled with cached tensors when you run data extraction code.

Make sure to instal this data in the data folder and with the appropriate folder names as specified by the constants in `data_extraction.py`.

## Local Setup

MacOS Sequoia 15.2 - Arm64 
Visual Studio Code
Python 3.10.16

Find pip dependences in `requirements.txt`


To train the CNN, simply modify the model structure in TrainingModel.__init__ and then run the `train_cnn.ipynb` notebook after changing `MODEL_NAME` and setting `training_params` approporiately.

To test the CNN in `test_models.ipynb`, simply change `MODEL_NAME` to the name of the model you trained in `train_cnn.ipynb` and then set `CHOSEN_MODEL` to `my_image_model`. You can also change the video paths in the function calls to `predict_on_video` to the videos you want to run the model on. If you want to evaluation on the test
dataset from `train_cnn.ipynb`, run `evaluate_cnn.ipynb` with `MODEL_NAME` set to the model you made in `train_cnn.ipynb`. This will access the tensors you saved in that training run, so make sure they are present before running it.

To test the optical flow classifier in `test_models.ipynb` , simply change  `CHOSEN_MODEL` to `optical_flow_model` and run the notebook as described above.

Note that for testing either of these models with `test_models.ipynb` you will need to adjust the path of videos that
are given in the `predict_on_video` function call. you can simply pick arbitrary videos from the `fire_videos` dataset you installed, or use vides of your choice.

To visualize the videos and data used in training and predicting, simply run `test_data_extraction.ipynb` after confirming that the paths in `data_extraction.py` point to the right directories and have data in them.


## Demos

Here is a [link](https://drive.google.com/drive/folders/1GLSxCgdm0lU-UBnkRTfptQrsf-lBjziE?usp=drive_link) to videos the models' predictions on a set of videos found of fires and non-fires found on youtube. The cnn folder contains predictions from the cnn, and the optical_flow folder contains predictions from the optical flow model.


## Presentation

Here is a link to a presentation on this project.