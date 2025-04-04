"""
Adithya Palle
Mar 31, 2025
Final Project

Contains functionality for training CNN models for fire detection from images.
"""

import random
from cnn import CNNFireDetector
import torch
from data_extraction import FireDataset, TARGET_IMAGE_SIZE, TRANSFORM
import dataclasses
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2




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

        conv_channels = 64
        conv_kernel_size = 3
        conv_padding = 0
        conv_stride = 1
        pooling_kernel_size = 3
        pooling_stride = 2

        final_conv_feature_maps = 8
        dropout_prob = 0.5

        fc_nodes = 64
        # TODO: rearchitectu model to use dropout, and not be a copy of source 150 
        # things to try: - skip connections, residual connections, batch normalization, dropout
        # change train /test/val to 80/10/10
        # try different hyperparameters
        # try not to shrink all the way to 1 feature map, have more fully connected nodes

        self.convolutional_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Conv2d(conv_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Conv2d(conv_channels, final_conv_feature_maps, kernel_size=conv_kernel_size, stride=conv_stride, padding = conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Flatten()
        )

        # calculate number of values in the flattened output of the convolutional layers
        conv_layers_output_size = calculate_conv_output_size(self.convolutional_layers, (3, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

        self.sequential_layers = torch.nn.Sequential(
            torch.nn.Linear(conv_layers_output_size, fc_nodes),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(fc_nodes, fc_nodes),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=dropout_prob),
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

        Args:
            x : image tensor or batch of image tensors, should be preprocessed before passing to the model
        Returns:
            float for fire probability between 0 and 1
        """
    
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
    

    


def train_cnn(model_and_transform :ModelWithTransform ,
              training_parameters : TrainingParameters,
              train_image_data : FireDataset, 
              validation_image_data: FireDataset) -> Tuple[XYData, XYData, CNNFireDetector]:
    """
    Trains a CNN model for fire detection with the given training data.
    Args:
        model (torch.nn.Module): The CNN model to train.
        training_parameters (TrainingParameters): The training hyperparameters.
        train_image_data (FireDataset): The training data.
        validation_image_data (FireDataset): The validation data used to adjust hyperparameters.
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
    train_dataloader = torch.utils.data.DataLoader(train_image_data, batch_size=training_parameters.batch_size, shuffle=True)

    # validation data is a single batch with the entire epoch
    val_dataloader = torch.utils.data.DataLoader(validation_image_data, batch_size=len(validation_image_data), shuffle=False)

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

    model = CNNFireDetector(inference_model, 
                            device = training_parameters.device, 
                            transform = transform)

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


def visualize_layer_weights(layer : torch.nn.Module , img : torch.Tensor, max_filters : int = 8):
    """
    Visualize the first layer weights of the model.
    Args:
        model (TrainingModel): The model to visualize.
        img (torch.Tensor): The image to visualize the weights on (must be preprocessed to be able to fed directly to layer).
        max_filters (int): The maximum number of filters to visualize.
    
    """
    # get module twice cause its a sequential nested in another sequential
    filters = layer.weight  # shape (num_filters, 3, 3, 3)
    filters = torch.stack(random.sample(list(filters), max_filters))  # random sample of filters so that we have max_filters


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
            larger_img = cv2.resize(filtered_imgs[i].cpu().numpy(), (400, 400), interpolation=cv2.INTER_LINEAR)
            axes[row, col].imshow(larger_img, cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle("Application of CNN Filters to Image", fontsize=16, fontweight="bold")
        plt.show()    

    
def visualize_layer_output(layer:torch.nn.Module, raw_img : torch.Tensor, transform : torch.nn.Module, device : torch.device = torch.device("cpu")):
    """
    Visualize the output of a layer in the model on a given image.
    First show the original image, then apply the transform to the image and visualize the output of the layer.
    Args:
        layer (torch.nn.Module): The layer to visualize.
        raw_img (torch.Tensor): The image to visualize the output on (must be preprocessed to be able to fed directly to layer).
        transform (torch.nn.Module): The transform to apply to the image before passing it to the layer.

    """
    # show original image
    img = raw_img.cpu().permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Image")
    plt.show()
    # apply transform to image and visualize the output of the layer
    visualize_layer_weights(layer, transform(raw_img).to(device), max_filters=8)
