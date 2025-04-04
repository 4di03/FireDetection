"""
Adithya Palle
Mar 31, 2025
Final Project

Contains class defining a custom CNN model for fire detection from images.
"""

from image_model import ImageModel
from data_extraction import TRANSFORM
import torch

class CNNFireDetector(ImageModel):
    """
    custom CNN image model based on pytorch nn.Module
    """

    def __init__(self, model : torch.nn.Module,
                  device : torch.device = torch.device("cpu"), 
                  transform : torch.nn.Module = None):
        """
        Initialize the CNNFireDetector with a pre-trained model.
        
        Args:
            model: A pre-trained CNN model for fire detection. It must be responsible for its own image preprocessing in it's forward pass.
            device: The device to run the model on (CPU or GPU). Default is CPU.
            transform: A transformation to be applied to the input image. Default is None.
        """
        self.device = device
        self.model = model.to(device)  # Move the model to the specified device
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transform  # Store the transformation if provided


    def predict(self, image : torch.tensor) -> float:
        """
        Predict the fire probability for the given image.

        Args:
            image: The input image to be processed. Will have dimensions (3, H, W). We need to preprocess as it is not preprocessed
        Returns:
            A float representing the fire probability for the entire image.
        """        
        image = image.to(self.device)  # Move the image to the specified device
        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)  # Add batch dimension
        else:
            print("No transform provided")
        with torch.no_grad():
            return self.model(image).item()
    

    def save_to_file(self, filename: str):
        """
        Save the model to a file.
        
        Args:
            filename: The name of the file to save the model to.
        """
        torch.save(self.model.state_dict(), filename)

    @staticmethod
    def load_from_file(filename: str, model : torch.nn.Module,  **kwargs) -> "CNNFireDetector":
        """
        Load the model from a file.
        """
        # load model on cpu
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        return CNNFireDetector(model, **kwargs)
    