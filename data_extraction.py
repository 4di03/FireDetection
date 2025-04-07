"""
Adithya Palle
Mar 31, 2025
Final Project

File containing functionality to load image and video data or training and evaluating models.
"""

from typing import Tuple, List
import torch
import torchvision.io
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import random
from torch.utils.data import Dataset

PLACES_DATA_PATH = "data/data_256"
FIRE_VIDEOS_DATA_PATH = "data/fire_videos"
FIRE_IMAGE_DATA_PATH = "data/Fire_Detection.v1.coco/train/"
TENSOR_CACHE_PATH = "data/tensor_cache"

# preprocessing transform to apply to the images before passing to the model. Does not change the number of dimensions of the image
TRANSFORM = transforms.Compose([ 
                                # conver to float tensor in range [0,1]
                                transforms.Lambda(lambda x: x.float()/255.0),
                                # resize to 224x224
                                transforms.Lambda(lambda x: 
                                                  F.interpolate(x.unsqueeze(0), 
                                                                size=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), 
                                                                mode='bilinear', 
                                                                align_corners=False).squeeze(0)),
                                # normalize based on ImageNet mean and std
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])


# size that we resize the images (square) to before passing to the model
TARGET_IMAGE_SIZE = 128


class FireDataset(Dataset):
    """
    Class to hold image dataset
    Attributes:
        image_data (torch.Tensor): Tensor representing the image which has been preprocessed before hand (B, C, H, W).
        is_fire (torch.Tensor): tensor of 1s and 0s representing fire (1) or no fire (0) of length B.
    """
    images: torch.Tensor
    labels: torch.Tensor

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images (torch.Tensor): Tensor representing the image which has been preprocessed before hand (B, C, H, W).
            labels (torch.Tensor): tensor of 1s and 0s representing fire (1) or no fire (0) of length B.
        """
        self.images = images
        self.labels = labels
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("The number of images and labels must be the same.")

    @staticmethod
    def init_from_data_name(data_name : str) -> "FireDataset":
        """
        Initalize a datset given the name of the tensors for images and labels.
        Args:
            data_name (str): The name of the dataset to load.
        Returns:
            FireDataset
        """
        images = torch.load(os.path.join(TENSOR_CACHE_PATH, f"{data_name}_images.pt"))
        labels = torch.load(os.path.join(TENSOR_CACHE_PATH, f"{data_name}_labels.pt"))
        return FireDataset(images, labels)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns the image and label at the given index.
        Args:
            index (int): The index of the image and label to return.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and the label tensor.    
        """
        return self.images[index], self.labels[index]

    def save(self, file_name: str):
        """
        Save the dataset to a file.
        Args:
            file_name (str): The name of the file to save the dataset to.
        """
        torch.save(self.images, file_name + "_images.pt")
        torch.save(self.labels, file_name + "_labels.pt")

class VideoData:
    """
    Class to hold video data
    Attributes:
        video_data (List[torch.Tensor]): List of tensors representing videos.
        is_fire (bool): True if fire, False if no fire.
    """
    video_data: List[torch.Tensor]
    is_fire: bool


# video data which is a list of tensors (each tensor is a video) and a boolean (True if fire, False if no fire) for each video
VideoData = List[torch.Tensor]

def load_frames_from_video(video_path : str) -> torch.Tensor:
    """
    Reads a tensor of frames from a video at a given path.
    Args:
        video_path (str) : the file path of the video to read
    Returns:
        frames (torch.Tensor) : a tensor of shape [T, C, H, W] representing the video data
    """
    frames, _, _ = torchvision.io.read_video(video_path)  # frames: [T, H, W, C]
    # resize image to [T, C, H , W]
    return frames.permute([0,3,1,2])

def get_video_data(videos_path : str =FIRE_VIDEOS_DATA_PATH + "/validation" ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads videos from directory and returns a tuple of tensors containing image data.
    Args:
        videos_path (str): The path to the directory containing the videos.
        The directory should contain two subdirectories: "pos" and "neg" in which pos contains videos with fire and neg contains videos without fire.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two lists of tensors:
            - The first list contains tensors representing videos with fire.
            - The second list contains tensors representing videos without fire.
    """

    pos_videos_path = os.path.join(TENSOR_CACHE_PATH, "pos_videos.pt")
    neg_videos_path = os.path.join(TENSOR_CACHE_PATH, "neg_videos.pt")
    # check tensor cache for cached tensors
    if  os.path.isfile(pos_videos_path) and os.path.isfile(neg_videos_path):
        pos_videos = torch.load(pos_videos_path)
        neg_videos = torch.load(neg_videos_path)
    else:
        pos_videos = []
    
        for pos_video in os.listdir(os.path.join(videos_path, "pos")):
            pos_video_path = os.path.join(videos_path, "pos", pos_video)

            frames =load_frames_from_video(pos_video_path)
            pos_videos.append(frames)  # Append the frames tensor to the list
    
        neg_videos = []
        for neg_video in os.listdir(os.path.join(videos_path, "neg")):
            neg_video_path = os.path.join(videos_path, "neg", neg_video)
            frames =load_frames_from_video(neg_video_path)
            
            neg_videos.append(frames)  # Append the frames tensor to the list
            
        # save to tensor cache
        torch.save(pos_videos, pos_videos_path)
        torch.save(neg_videos, neg_videos_path)
    

    return pos_videos, neg_videos

def display_video(video_tensor: torch.Tensor, num_frames: int = 5):
    """
    Display the first num_frames of a video tensor
    each video tensor should be (T, H, W, C) shape
    Args:
        video_tensor (torch.Tensor): The video tensor to display.
        num_frames (int): The number of frames to display.
    
    """
    num_frames = min(num_frames, video_tensor.shape[0])
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i in range(num_frames):
        frame = video_tensor[i] #
        axes[i].imshow(frame.permute(1, 2, 0).numpy())  # Convert to HWC format for display
        axes[i].axis('off')
    plt.show()


def get_random_img_file_data(directory, n) -> List[torch.Tensor]:
    """
    Get n random jpg files from a directory and its subdirectories.
    Args:
        directory (str): The directory to search for jpg files.
        n (int): The number of random jpg files to return.
    Returns:
        List[torch.Tensor]: A list of n random jpg images converted to tensors.
    """
    jpg_files = []
    print("Searching for jpg files in directory:", directory)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().split(".")[-1] in ["jpg", "jpeg", "png"]:
                jpg_files.append(os.path.join(root, file))
    print(f"Found {len(jpg_files)} jpg files in directory: {directory}")
    
    # randomly sample n jpg file paths and then read the file as tensor
    image_tensors = list(map(lambda img_path : torchvision.io.read_image(img_path), 
                             random.sample(jpg_files, min(n, len(jpg_files)))))
    return image_tensors

ImageData = List[Tuple[torch.Tensor, bool]]


def preprocess_data(image_data: ImageData) -> Tuple[FireDataset, FireDataset, FireDataset]:
    """
    Preprocess the image data and split it into train, validation, and test sets.
    Args:
        image_data (ImageData): The image data to preprocess.
    Returns:
        Tuple[FireDataset, FireDataset, FireDataset]: A tuple containing the train, validation, and test datasets.
    """


    all_images = image_data

    n_total_samples = len(all_images)
    # split into train, val, test
    n_train_samples = int(n_total_samples * 0.8)
    n_val_samples = int(n_total_samples * 0.1)
    train_images = all_images[:n_train_samples]
    val_images = all_images[n_train_samples:n_train_samples + n_val_samples]
    test_images = all_images[n_train_samples + n_val_samples:] # will have remaining 10% of data since we are using 90% for train and val


    # stack the images and labels
    float_train_labels = [float(sample[1]) for sample in train_images]
    float_val_labels = [float(sample[1]) for sample in val_images]
    float_test_labels = [float(sample[1]) for sample in test_images]

    train_images = torch.stack([TRANSFORM(sample[0]) for sample in train_images])
    train_labels = torch.tensor(float_train_labels)
    val_images = torch.stack([TRANSFORM(sample[0]) for sample in val_images])
    val_labels = torch.tensor(float_val_labels)
    test_images = torch.stack([TRANSFORM(sample[0]) for sample in test_images])
    test_labels = torch.tensor(float_test_labels)

    return FireDataset(train_images, train_labels), FireDataset(val_images, val_labels), FireDataset(test_images, test_labels) 

def get_image_data(n_total_samples = 1000) -> ImageData:
    """
    Generate random data for training and testing. Apply preprocessing to the images as well using TRANSFORM.
    60/20/20 split for train/validation/test
    Args:
        n_total_samples (int): Total number of samples to generate.
    Returns:
        ImageData: A list of tuples containing the unprocessed image tensors and their corresponding labels.
    """

    

    # get complete dataset (random sample of n_total_samples images) such that 50% are fire and 50% are no fire
    num_fire_images = n_total_samples // 2

    fire_jpg_files = get_random_img_file_data(FIRE_IMAGE_DATA_PATH, num_fire_images)
    num_fire_images = len(fire_jpg_files)

    fire_images =  list(zip(fire_jpg_files, [True] * num_fire_images))
    num_nofire_images = num_fire_images

    print(f"Number of fire images: {num_fire_images}")
    print(f"Number of no fire images: {num_nofire_images}")

    nofire_images = list(zip(get_random_img_file_data(PLACES_DATA_PATH, num_nofire_images), [False] * (num_nofire_images)))

    # combine the lists and shuffle them
    all_images = fire_images + nofire_images
    random.shuffle(all_images)

    return all_images


def display_images(image_data: FireDataset, num_images: int = 5, title = "Random Sample of Images"):
    """
    Display random sample of a list of image tensors
    Args:
        image_data (FireDataset): List of image tensors.
        num_images (int): Number of images to display.
        title (str): Title for the plot.
    
    """
    num_images = min(num_images, len(image_data))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # sample without replacement
    random_indices = random.sample(range(len(image_data)), num_images)

    # add title  to entire figure
    fig.suptitle(title)
    # remove white space between title and images
    fig.subplots_adjust(top=1.2)

    for i in range(num_images):
        img_tensor, is_fire = image_data[random_indices[i]]
        axes[i].imshow(torchvision.transforms.functional.to_pil_image(img_tensor))
        axes[i].set_title("Fire" if is_fire else "No Fire")
        axes[i].axis('off')
    plt.show()