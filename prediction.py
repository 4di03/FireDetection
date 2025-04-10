"""
Adithya Palle 
March 30, 2025 
Final Project

File containg functions for running predictions on videos and getting performance metrics.
"""
import torch 
from typing import List, Tuple
from video_model import VideoModel
from train_cnn import InferenceModel
import time

def get_metrics(predictions : List[float], labels : List[float], threshold : float) -> Tuple[float,float, float]:
    """
    Get metrics for the model predictions using the given threshold.
    Args:
        predictions (List[float]): List of predictions for each image
        labels (List[float]): List of labels for each image (1 for fire, 0 for no fire)
        threshold (float): The threshold to use for determining fire.
        
    Returns:
        float: The recall for the model predictions.
        float: The false positive rate for the model predictions.
        float: The accuracy for the model predictions.
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    total = len(predictions)
    
    for i in range(total):
        if predictions[i] >= threshold and labels[i] == 1:
            tp += 1
        elif predictions[i] < threshold and labels[i] == 0:
            tn += 1
        elif predictions[i] >= threshold and labels[i] == 0:
            fp += 1
        elif predictions[i] < threshold and labels[i] == 1:
            fn += 1

    recall = tp / (tp + fn)
    accuracy = (tp + tn) / total
    false_positive_rate = fp / (fp + tn)

    return recall,false_positive_rate, accuracy



def get_predictions_on_videos(model : VideoModel, videos : List[torch.Tensor]) -> Tuple[List[List[float]], float, int]:
    """
    Get fire probility predictions for each frame in the video.
    Args:
        model (VideoModel): The model to use for predictions.
        videos (List[torch.Tensor]): List of videos to predict on.
    Returns:
        List[List[float]]: List of predictions for each frame in the video.
        float : the total time to predict on all videos (ms)
        int : the total number of frames in all videos
    """
    predictions = []
    total_time_ms = 0
    total_frames = 0
    for video in videos:
        start = time.perf_counter()
        pred = model.predict_on_full_video(video)
        time_taken = (time.perf_counter() - start) * 1000  # convert to ms
        total_time_ms += time_taken
        total_frames += len(video)
        predictions.append(pred)

    return predictions, total_time_ms, total_frames


def get_recall(predictions_on_fire_video : List[List[float]], threshold : float) -> float:
    """
    Get recall for the model predictions using the given threshold.
    Args:
        predictions (List[List[float]]): List of predictions for each frame in a video where all frames are positives.
        threshold (float): The threshold to use for determining fire.
    Returns:
        float: The recall for the model predictions.
    """
    tp = 0
    fn = 0
    for pred in predictions_on_fire_video:
        for p in pred:
            if p >= threshold:
                tp += 1
            else:
                fn += 1

    return tp / (tp + fn)


def get_accuracy(predictions_on_fire_video : List[List[float]],predictions_on_nofire_video : List[List[float]], threshold : float) -> float:
    """
    Get accuracy for the model predictions using the given threshold.
    Args:
        predictions_on_fire_video (List[List[float]]): List of predictions for each frame in a video where all frames are positives.
        predictions_on_nofire_video (List[List[float]]): List of predictions for each frame in a video where all frames are negatives
        threshold (float): The threshold to use for determining fire.
    Returns:
        float: The accuracy for the model predictions. (True Positives + True Negatives) / (all predictions)
    """

    total = 0
    tp = 0
    tn = 0
    for pred in predictions_on_fire_video:
        for p in pred:
            if p >= threshold:
                tp += 1 # correct if the prediction is fire
        total += len(pred)
    for pred in predictions_on_nofire_video:
        for p in pred:
            if p < threshold:
                tn += 1 # correct if the prediction is no fire
        total+= len(pred)
    return (tp + tn) / total
def get_false_positive_rate(predictions_on_nofire_video : List[List[float]], threshold : float) -> float:
    """
    Get false positive rate for the model predictions using the given threshold.
    Args:
        predictions (List[List[float]]): List of predictions for each frame in a video where all frames are negatives.
        threshold (float): The threshold to use for determining fire.
    Returns:
        float: The false positive rate for the model predictions.
    """
    fp = 0
    tn = 0
    for pred in predictions_on_nofire_video:
        for p in pred:
            if p >= threshold:
                fp += 1
            else:
                tn += 1

    return fp / (fp + tn)

