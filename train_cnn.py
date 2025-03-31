"""
Adithya Palle
Mar 31, 2025
Final Project

Contains functionality for training CNN models for fire detection from images.
"""

from cnn import CNNFireDetector
import torch
from data_extraction import ImageData, get_image_data
import sys
import dataclasses
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

# preprocessing step for images
TARGET_IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([ # resize to 224x224
                                transforms.Lambda(lambda x: 
                                                  F.interpolate(x.unsqueeze(0), 
                                                                size=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), 
                                                                mode='bilinear', 
                                                                align_corners=False).squeeze(0)),
                                # normalize based on imagenet mean and std
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

# TODO: consider recalcualating mean and std for the dataset if these do not work well

def init_torch_model() -> torch.nn.Module:
    """
    Factory method for a CNN used for binary classification for fire detection.
    The model should not output sigmoid at the end since we use BCEWithLogitsLoss, this is purely a training model
    and the actual inference model is a sequential model with the sigmoid at the end created in the train_cnn function.
    Returns:
        torch.nn.Module: A PyTorch model for fire detection.
    """
    size_after_first_conv = (TARGET_IMAGE_SIZE - 3 + 1)
    size_after_first_pool = size_after_first_conv // 2
    size_after_second_conv = (size_after_first_pool - 3 + 1)
    size_after_second_pool = size_after_second_conv // 2
    size_after_flatten = size_after_second_pool * size_after_second_conv

    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=1),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(size_after_flatten, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1) # dont use sigmoid at the end since we use BCEWithLogitsLoss
    )

@dataclasses.dataclass
class TrainingParameters:
    optimizer : torch.optim.Optimizer
    loss_function : torch.nn.Module
    batch_size : int
    n_epochs : int
    early_stopping_threshold : float = 0.01 # threshold for which we stop trianing if loss is less change is less than this for consecutive epochs where loss is decreasing


@dataclasses.dataclass
class ModelWithTransform:
    """
    Dataclass to store a model and its transform (preprocessing that is applied to images before passing to the model)
    """
    model : torch.nn.Module
    transform : torch.nn.Module


def get_total_avg_loss(network : torch.nn.Module, 
                       dataloader : torch.utils.data.DataLoader, 
                       loss_function : torch.nn.Module) -> float:
    """
    Computes the average loss on the entire dataset using the given network and data loader.

    Args:
        network: the network to use
        dataloader: the data loader to use
        loss_function: the loss function
    Returns:
        float: the loss on the entire dataset, averaged over all batches
    """
    total_batches = len(dataloader)

    if total_batches == 0:
        raise ValueError("Dataloader is empty")
    total_loss = 0
    network.eval()  # set network to evaluation mode

    with torch.no_grad():
        for test_images, test_labels in dataloader:
            test_output = network(test_images)
            total_loss += loss_function(test_output, test_labels).item()

    return total_loss / total_batches   
@dataclasses.dataclass
class XYData:
    """
    Dataclass to store x and y values for plotting
    """
    x: List[float] # x values
    y: List[float] # y values


    def get_final_loss(self):
        """
        Returns the loss of the maximal x point
        Args:
            None
        """
        return self.y[self.x.index(max(self.x))]
    


class FireDataset(Dataset):
    """ Custom dataset for binary classification from images"""
    def __init__(self, data: ImageData, transform : torch.nn.Module = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image, label = self.data[idx]

        # apply preprocessing to the image
        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(int(label), dtype=torch.float32)  # or long if using CrossEntropyLoss
        return image, label_tensor
    


def train_cnn(model_and_transform :ModelWithTransform ,
              training_parameters : TrainingParameters,
              train_image_data : ImageData, 
              validation_image_data: ImageData) -> Tuple[XYData, XYData, CNNFireDetector]:
    """
    Trains a CNN model for fire detection with the given training data.
    Args:
        model (torch.nn.Module): The CNN model to train.
        training_parameters (TrainingParameters): The training hyperparameters.
        train_image_data (ImageData): The training data.
        validation_image_data (ImageData): The validation data used to adjust hyperparameters.
    Returns:
        Tuple[XYData, XYData, CNNFireDetector]: A tuple containing:
            - train_loss_plot: The training loss data.
            - val_loss_plot: The validation loss data.
            - model: The trained CNNFireDetector model.
    """


    print("Begin training...")
    train_losses = []
    val_losses = []
    model = model_and_transform.model
    transform = model_and_transform.transform

    # Create data loaders for training and validation data
    train_dataloader = torch.utils.data.DataLoader(FireDataset(train_image_data, transform), batch_size=training_parameters.batch_size, shuffle=True)

    # validation data is a single batch with the entire epoch
    val_dataloader = torch.utils.data.DataLoader(FireDataset(validation_image_data, transform), batch_size=len(validation_image_data), shuffle=False)

    # Train the network
    for epoch_index in range(training_parameters.n_epochs):  # loop over the dataset multiple times
        model.train()  # set network to training mode

        cur_epoch_train_loss = 0
        for batch_index, (train_images, train_labels) in enumerate(train_dataloader):
            training_parameters.optimizer.zero_grad()  # zero the gradients for safety


            output = model(train_images)  # forward pass on training data

            train_loss = training_parameters.loss_function(output, train_labels)  # calculate loss
            train_loss.backward()  # backpropagation
            training_parameters.optimizer.step()  # update weights

            cur_epoch_train_loss += train_loss.item()

        
        # store average train loss for the epoch
        cur_epoch_train_loss /= len(train_dataloader)
        train_losses.append(cur_epoch_train_loss)
        


        # predict on validation data after each epoch
        val_losses.append(get_total_avg_loss(model, val_dataloader, training_parameters.loss_function))

        print(f"Epoch {epoch_index + 1} completed. Train loss: {cur_epoch_train_loss:.4f}, Validation loss: {val_losses[-1]:.4f}")

        # compare with previous val loss for early stopping
        if epoch_index > 0 and  val_losses[-1] < val_losses[-2] and val_losses[-2] - val_losses[-1] < training_parameters.early_stopping_threshold:
            print("Early stopping")
            break


    train_loss_plot = XYData(x=range(len(train_losses)), y=train_losses)
    val_loss_plot = XYData(x=range(len(val_losses)), y=val_losses)

    inference_model = torch.nn.Sequential(
        model,
        torch.nn.Sigmoid() # apply sigmoid to the output for binary classification
    )

    model = CNNFireDetector(inference_model)

    return train_loss_plot, val_loss_plot, model


def init_training_params(model : torch.nn.Module) -> TrainingParameters:
    """
    Get default training parameters for the CNN model.
    Returns:
        TrainingParameters: The training parameters loaded from the file.
    """
    
    return TrainingParameters(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        loss_function=torch.nn.BCEWithLogitsLoss(), # Binary Classification Loss that automatically applies sigmoid
        batch_size=32,
        n_epochs=10,
        early_stopping_threshold=0.01
    )


def visualize_loss_curve(training_loss : XYData, val_loss : XYData):
    """
    Visualize the training and validation loss curves.
    Args:
        training_loss (XYData): The training loss data.
        val_loss (XYData): The validation loss data.
    """
    if len(training_loss.x) == 0 or len(val_loss.x) == 0:
        print("No data to plot")
        return
    

    plt.plot(training_loss.x, training_loss.y, label='Training Loss')
    plt.plot(val_loss.x, val_loss.y, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_cnn.py <n_samples> <output_file_path>")
        sys.exit(1)



    n_samples = int(sys.argv[1])
    output_file_path = sys.argv[2]

    print(f"Using {n_samples} images for training and testing.")

    model = init_torch_model()
    training_params = init_training_params(model)

    # resize to 224x224 as preprocessing step
    model_and_transform = ModelWithTransform(model, TRANSFORM)

    train,val,test = get_image_data(n_samples)

    train_loss, val_loss ,fire_detector = train_cnn(model_and_transform,training_params,train, val)

    visualize_loss_curve(train_loss, val_loss)
    # Save the trained model to a file
    fire_detector.save_to_file(output_file_path)
    print(f"Model saved to {output_file_path}")


