"""
Adithya Palle 
March 30, 2025 
Final Project

File containg functions for running predictions on videos and getting performance metrics.
"""
import torch 
from typing import List
from video_model import VideoModel

def get_predictions(model : VideoModel, videos : List[torch.Tensor]) -> List[List[float]]:
    """
    Get fire probility predictions for each frame in the video.
    Args:
        model (VideoModel): The model to use for predictions.
        videos (List[torch.Tensor]): List of videos to predict on.
    Returns:
        List[float]: List of predictions for each frame in the video.
    """
    predictions = []
    for video in videos:
        predictions.append(model.predict_on_full_video(video))

    return predictions


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

