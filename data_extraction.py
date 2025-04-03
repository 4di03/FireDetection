from typing import Tuple, List
import torch
import torchvision.io
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random


#DFIRE_DATA_PATH = "data/D-Fire"
PLACES_DATA_PATH = "data/data_256"
#FIRE_DATA_PATH = "data/fire_dataset"
FIRE_VIDEOS_DATA_PATH = "data/fire_videos"
FIRE_IMAGE_DATA_PATH = "data/Fire_Detection.v1.coco/train/"
TENSOR_CACHE_PATH = "data/tensor_cache"

# image data which is an image tensor and a boolean (True if fire, False if no fire) for each image
ImageData = List[Tuple[torch.Tensor, bool]]

# video data which is a list of tensors (each tensor is a video) and a boolean (True if fire, False if no fire) for each video
VideoData = List[torch.Tensor]

def save_image_data(data : ImageData, file_name : str):
    """
    Store images and label tensors at a path with the given file name

    Args:
        data (ImageData) : the list of tuple of images and labels
        file_name (str) : prefix of file path where you will save the images and labels
    
    """
    stacked_features = torch.stack([sample[0] for sample in data])
    stacked_labels = torch.tensor([sample[1] for sample in data , dtype=torch.float32)

    features_fp = f"{stacked_features}_imgs.pt"
    labels_fp = f"{stacked_labels}_labels.pt"
    
    torch.save(stacked_features, features_fp)
    torch.save(stacked_labels, labels_fp)



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
            frames, _, _ = torchvision.io.read_video(pos_video_path)  # frames: [T, H, W, C]
            # resize image to [T, C, H , W]
            frames = frames.permute([0,3,1,2])
    
            pos_videos.append(frames)  # Append the frames tensor to the list
    
        neg_videos = []
        for neg_video in os.listdir(os.path.join(videos_path, "neg")):
            neg_video_path = os.path.join(videos_path, "neg", neg_video)
            frames, _, _ = torchvision.io.read_video(neg_video_path)  # frames: [T, H, W, C]
            # resize image to [T, C, H , W]
            frames = frames.permute([0,3,1,2])
            
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
        axes[i].imshow(frame)
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

def get_image_data(n_total_samples = 1000) -> Tuple[ImageData, ImageData, ImageData]:
    """
    Generate random data for training and testing.
    60/20/20 split for train/validation/test
    Args:
        n_total_samples (int): Total number of samples to generate.
    Returns:
        Tuple[ImageData, ImageData, ImageData]: Tuple of train, validation, and test data.
        Each data set is a list of tuples, where each tuple contains an image tensor and a boolean (True if fire, False if no fire).
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

    n_total_samples = len(all_images)
    # split into train, val, test
    n_train_samples = int(n_total_samples * 0.8)
    n_val_samples = int(n_total_samples * 0.1)
    train_images = all_images[:n_train_samples]
    val_images = all_images[n_train_samples:n_train_samples + n_val_samples]
    test_images = all_images[n_train_samples + n_val_samples:] # will have remaining 10% of data since we are using 90% for train and val

    return train_images, val_images, test_images


def display_images(image_data: ImageData, num_images: int = 5, title = "Random Sample of Images"):
    """
    Display random sample of a list of image tensors
    Args:
        image_data (ImageData): List of image tensors.
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
        axes[i].imshow(F.to_pil_image(img_tensor))
        axes[i].set_title("Fire" if is_fire else "No Fire")
        axes[i].axis('off')
    plt.show()