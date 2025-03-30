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
            image: The input image to be processed.

        Returns:
            A float representing the fire probability for the entire image.
        """
        pass



