from video_model import VideoModel
import cv2
import numpy as np




class FarnebackOpticalFlowModel(VideoModel):


    def predict_on_last_frame(self, frames) -> float:
        """
        Predicts the fire probability for the last frame in the given stream of images using Farneback Optical Flow.
        This takes the last two frames and computes the optical flow between them.
        It then determines the flow vectors for color regions that match fire (orange-red)
        produces a fire probability between 0 and 1 based on the sumemd flow in those regions.
        Args:
            frames: A list of tensors representing the frames in the video stream. The last frame is the most recent one for which the prediction is made.
        Returns:
            A float representing the fire probability for the last frame in the video stream.
        
        """
        pass


