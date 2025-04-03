"""
Adithya Palle 
March 30, 2025 
Final Project

File containing interface for image models that predict fire probability for a given image.
"""
import torch
from abc import ABC, abstractmethod

class ImageModel(ABC):
    """
    Interface for a model that takes a single image as input and returns a fire/no-fire probability prediction for the entire image.
    """

    @abstractmethod
    def predict(self, image : torch.tensor) -> float:
        """
        Predicts the fire probability for the given image.

        Args:
            image: The input image to be processed. This image is not preprocessed.

        Returns:
            A float representing the fire probability for the entire image.
        """
        pass



