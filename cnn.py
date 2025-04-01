"""
Adithya Palle
Mar 31, 2025
Final Project

Contains class defining a custom CNN model for fire detection from images.
"""

from image_model import ImageModel
import torch

class CNNFireDetector(ImageModel):
    """
    custom CNN image model based on pytorch nn.Module
    """

    def __init__(self, model : torch.nn.Module, device : torch.device = torch.device("cpu")):
        """
        Initialize the CNNFireDetector with a pre-trained model.
        
        Args:
            model: A pre-trained CNN model for fire detection. It must be responsible for its own image preprocessing in it's forward pass.
        """
        self.model = model.to(device)  # Move the model to the specified device
        self.model.eval()  # Set the model to evaluation mode


    def predict(self, image : torch.tensor) -> float:
        """
        Predict the fire probability for the given image.

        Args:
            image: The input image to be processed. Will have dimensions (3, H, W). The model is responsible for preprocessing it as needed in its forward() function.
        Returns:
            A float representing the fire probability for the entire image.
        """        
        return self.model(image).item()
    

    def save_to_file(self, filename: str):
        """
        Save the model to a file.
        
        Args:
            filename: The name of the file to save the model to.
        """
        torch.save(self.model.state_dict(), filename)

    @staticmethod
    def load_from_file(model : torch.nn.Module, filename: str) -> "CNNFireDetector":
        """
        Load the model from a file.
        """
        model.load_state_dict(torch.load(filename))
        
        return CNNFireDetector(model)
    