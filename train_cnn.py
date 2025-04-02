"""
Adithya Palle
Mar 31, 2025
Final Project

Contains functionality for training CNN models for fire detection from images.
"""

from cnn import CNNFireDetector
import torch
from data_extraction import ImageData, get_image_data
import dataclasses
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
# size that we resize the images (square) to before passing to the model
TARGET_IMAGE_SIZE = 128


TRANSFORM = transforms.Compose([ 
                                # conver to float tensor in range [0,1]
                                transforms.Lambda(lambda x: x.float()/255.0),
                                # resize to 224x224
                                transforms.Lambda(lambda x: 
                                                  F.interpolate(x.unsqueeze(0), 
                                                                size=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), 
                                                                mode='bilinear', 
                                                                align_corners=False).squeeze(0)),
                                # normalize based on imagenet mean and std
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

# TODO: consider recalcualating mean and std for the dataset if these do not work well


def calculate_conv_output_size(model : torch.nn.Module , input_size : Tuple[int, int, int]) -> int:
    """
    Calculate the output size of the convolutional layers given an input size.
    Args:
        model: The model containing the convolutional layers.
        input_size: The size of the input tensor.
    Returns:
        int: The output size (number of values) after passing through the convolutional layers and flattening.
    """

    # make sure last layer is flatten
    if not isinstance(model[-1], torch.nn.Flatten):
        raise ValueError("Last layer of model must be a Flatten layer")
    

    # Create a dummy input tensor with the given size
    dummy_input = torch.randn(1, *input_size)  # (batch_size, channels, height, width)
    with torch.no_grad():
        # Pass the dummy input through the model
        output = model(dummy_input)
    
    # Get the size of the output after flattening
    output_size = output.numel()
        
    return output_size


class TrainingModel(torch.nn.Module):

    def __init__(self):
        """
        Initialize the TrainingModel with a pre-trained model.
        
        Args:
            model: A pre-trained CNN model for fire detection.
        """
        super(TrainingModel, self).__init__()

        # use leakyRelu to avoid dead neurons from negative inputs after normalization of the images

        conv_channels = 16
        conv_kernel_size = 5
        conv_padding = 0
        conv_stride = 1
        pooling_kernel_size = 3
        pooling_stride = 2
        fc_nodes = 128

        self.convolutional_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(conv_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(conv_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.LeakyReLU(),
            # reduce to 1 feature map to extract most important features
            torch.nn.Conv2d(conv_channels, 1, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten()
        )

        # calculate number of values in the flattened output of the convolutional layers
        conv_layers_output_size = calculate_conv_output_size(self.convolutional_layers, (3, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

        self.sequential_layers = torch.nn.Sequential(
            torch.nn.Linear(conv_layers_output_size, fc_nodes),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(fc_nodes, fc_nodes),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(fc_nodes, 1)
            # don't apply sigmoid here, we will do it in InferenceModel and instead we use BCEWithLogitsLoss for training
        )

    


        self.net = torch.nn.Sequential(
            self.convolutional_layers,
            self.sequential_layers
        )

        # print number of parameters in the model
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters in the model: {num_params}")

    def forward(self, x):
        return self.net(x)
    

class InferenceModel(torch.nn.Module):

    def __init__(self, trained_model : TrainingModel, transform : torch.nn.Module = TRANSFORM):
        """
        Initialize the InferenceModel with a pre-trained model.
        
        Args:
            model: A pre-trained CNN model for fire detection.
            transform : preprocessing transform to apply before predicting on images. Should be the same as the one used in training.
        """
        super(InferenceModel, self).__init__()
        self.trained_model = trained_model
        self.transform = transform
    def forward(self, x):
        """
        Predicts the fire probabilty for each image in x.
        We apply the approporiate preprocessing transform to x or the batch of images in x and then 
        pass it to the trained_model.

        TODO: consider batch preprpeocessing if this is too slow.
        Args:
            x : image tensor or batch of image tensors
        Returns:
            float for fire probability between 0 and 1
        """
        if len(x.shape) == 3:
            # Single image: (3, H, W)
            x = self.transform(x).unsqueeze(0)  # Make it (1, 3, H, W)
        elif len(x.shape) == 4:
            # Batch of images: (B, 3, H, W)
            x = torch.stack([self.transform(img) for img in x])
        else:
            raise ValueError("Expected input of shape (3, H, W) or (B, 3, H, W)")
    
        return torch.sigmoid(self.trained_model(x))




@dataclasses.dataclass
class TrainingParameters:
    optimizer : torch.optim.Optimizer
    loss_function : torch.nn.Module
    batch_size : int
    n_epochs : int
    scheduler : torch.optim.lr_scheduler = None # learning rate scheduler, default is None
    early_stopping_threshold : float = 1e-3 # threshold for which we stop trianing if loss is less change is less than this for consecutive epochs where loss is decreasing, set to 0 to disable
    device : torch.device = torch.device("cpu") # device to use for training, default is cpu

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
    device = next(network.parameters()).device
    with torch.no_grad():
        for test_images, test_labels in dataloader:

            # move data to device
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)


            test_output = network(test_images)
            total_loss += loss_function(test_output.squeeze(1), test_labels).item()

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
    model = model_and_transform.model.to(training_parameters.device)
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

            # move data to device
            train_images = train_images.to(training_parameters.device)
            train_labels = train_labels.to(training_parameters.device)

            training_parameters.optimizer.zero_grad()  # zero the gradients for safety


            output = model(train_images)  # forward pass on training data


            train_loss = training_parameters.loss_function(output.squeeze(1), train_labels)  # calculate loss
            train_loss.backward()  # backpropagation
            training_parameters.optimizer.step()  # update weights

            cur_epoch_train_loss += train_loss.item()

        
        # store average train loss for the epoch
        cur_epoch_train_loss /= len(train_dataloader)
        train_losses.append(cur_epoch_train_loss)
        


        # predict on validation data after each epoch
        val_loss = get_total_avg_loss(model, val_dataloader, training_parameters.loss_function)
        val_losses.append(val_loss)
        if training_parameters.scheduler is not None:
            # update the learning rate if using a scheduler
            training_parameters.scheduler.step(val_loss)  # step the scheduler if using one

            



        print(f"Epoch {epoch_index + 1} completed. Train loss: {cur_epoch_train_loss:.4f}, Validation loss: {val_losses[-1]:.4f}")

        # compare with previous val loss for early stopping
        if epoch_index > 0 and val_losses[-2] >=  val_losses[-1]  and (val_losses[-2] - val_losses[-1]) <= training_parameters.early_stopping_threshold:
            print(val_losses[-2] - val_losses[-1], training_parameters.early_stopping_threshold, (val_losses[-2] - val_losses[-1]) <= training_parameters.early_stopping_threshold)
            print("Early stopping")
            break


    train_loss_plot = XYData(x=range(len(train_losses)), y=train_losses)
    val_loss_plot = XYData(x=range(len(val_losses)), y=val_losses)

    inference_model = InferenceModel(model, transform)

    model = CNNFireDetector(inference_model)

    return train_loss_plot, val_loss_plot, model


def init_training_params(model : torch.nn.Module, device : torch.device) -> TrainingParameters:
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
        early_stopping_threshold=0,
        device = device
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


def visualize_layer_weights(layer : torch.nn.Module , img : torch.Tensor):
    """
    Visualize the first layer weights of the model.
    Args:
        model (TrainingModel): The model to visualize.
        img (torch.Tensor): The image to visualize the weights on.
    
    """
    # get module twice cause its a sequential nested in another sequential
    filters = layer.weight  # shape (num_filters, 3, 3, 3)
    print("Filters shape: ", filters.shape)
    print(filters.shape)

    img = img.unsqueeze(0)        # (1, 3, 244, 244) - add batch dimension
    
    with torch.no_grad():
        filtered_imgs = F.conv2d(img, filters, bias=None, stride=1, padding=1)  # (1, N, H, W)
        filtered_imgs = filtered_imgs.squeeze(0)  # (N, H, W)
        num_filters = filters.shape[0]

        # plot the filters and the filtered images
        fig, axes = plt.subplots(int(num_filters/4) ,4, figsize=(8, 8))
        for i in range(0,len(filters)):
            row, col = divmod(i, 4)

            # make filtered images bigger
            larger_img = cv2.resize(filtered_imgs[i].numpy(), (400, 400), interpolation=cv2.INTER_LINEAR)
            axes[row, col].imshow(larger_img, cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle("Application of CNN Filters to Image", fontsize=16, fontweight="bold")
        plt.show()    

    
