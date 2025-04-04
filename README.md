# FireDetection

## Group Members
Adithya Palle
## Data

To run the notebooks, you may need the appropriate data in your data folder. 

 TODO: add googel drive links to data


 After this, rearrange the files such that the paths in `data_extraction.py` point to the right directories and have data in them.

## Local Setup

MacOS Sequoia 15.2 - Arm64 
Visual Studio Code
Python 3.10.16


To train the CNN, simply modify the model structure in TrainingModel.__init__ and then run the `train_cnn.ipynb` notebook after changing `MODEL_NAME` 
and setting `training_params` approporiately.


To test the CNN in `test_models.ipynb`, simply change `MODEL_NAME` to the name of the model you trained in `train_cnn.ipynb` and then set `CHOSEN_MODEL` to `my_image_model`. You can also change the video paths in the function calls to `predict_on_video` to the videos you want to run the model on. If you want to evaluation on the test
dataset from `train_cnn.ipynb`, run `evaluate_cnn.ipynb` with `MODEL_NAME` set to that you made in `train_cnn.ipynb`. This will access the tensors you saved in that 
training run, so make sure they are present before running it.

To test the optical flow classifier in `test_models.ipynb` , simply change  `CHOSEN_MODEL` to `optical_flow_model` and run the notebook as described above.

To visualize the videos and data used in training and predicting, simply run `test_data_extraction.ipynb` after confirming that the paths in `data_extraction.py` point to the right directories and have data in them.


## Demos

TODO: add google drive  link to viedos of repdcets

## Presentation

# TODO:
- load in train / test / val distributions for join d-fire and places dataset
- get testing video feeds of fires for interfernce