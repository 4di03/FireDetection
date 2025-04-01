"""
Adithya Palle 
March 30, 2025 
Final Project

File containing Abstract class for image models that predict fire probability in the latest frame for a given stream of images.
"""
import torch
from abc import ABC, abstractmethod
from typing import List
from image_model import ImageModel

class VideoModel(ABC):
    """
    Abstract class for a model that takes a single image as input and returns a fire/no-fire probability prediction for the entire image.
    """

    @abstractmethod
    def predict_on_last_frame(self, frames : List[torch.tensor]) -> float:
        """
        Predicts the fire probability for the last frame in the given stream of images.
        Args:
            frames: A list of tensors representing the frames in the video stream. The last frame is the most recent one for which the prediction is made.

        Returns:
            A float representing the fire probability for the last frame in the video stream.
        """
        pass


    def predict_on_full_video(self, video : List[torch.tensor]) -> List[float]:
        """
        Predicts the fire probability for each frame in the given video stream.
        Args:
            video: A list of tensors representing the frames in the video stream.

        Returns:
            A list of floats representing the fire probability for each frame in the video stream.
        """
        predictions = []
        frames = []
        for frame in video:
            # build up the stream of frames
            frames.append(frame)
            
            prediction = self.predict_on_last_frame(frames)
            predictions.append(prediction)
        return predictions


class VideoModelFromImageModel(VideoModel):
    """
    A wrapper class that allows an image model to be used as a video model.
    """
    def __init__(self, image_model : ImageModel):
        self.image_model = image_model

    def predict_on_last_frame(self, frames : List[torch.tensor]) -> float:
        print("L59", frames[-1].shape)
        return self.image_model.predict(frames[-1]) # add batch dimension