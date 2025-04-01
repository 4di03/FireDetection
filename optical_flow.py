from video_model import VideoModel
import cv2
import numpy as np


FIRE_LOWER_HSV = np.array([0, 100, 100])
FIRE_UPPER_HSV = np.array([10, 255, 255])

class FarnebackOpticalFlowModel(VideoModel):

    # TODO: address weakness when there are lots of other moving obejcts in the scene (maybe feed to classifier such as LR)
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
        if len(frames) < 2:
            return 0.0
        # Convert the last two frames to grayscale
        prev_frame = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
        pred_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

        # Compute the optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, pred_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Calculate the magnitude and angle of the flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Create a mask for the fire color range (orange-red) in HSV
        # Convert the last frame to HSV color space
        hsv_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2HSV)
        # Create a mask for the fire color range
        fire_mask = cv2.inRange(hsv_frame, FIRE_LOWER_HSV, FIRE_UPPER_HSV)
        # Calculate the sum of the flow magnitudes in the fire regions
        flow_magnitude_in_fire = np.sum(magnitude[fire_mask > 0])
        # Calculate the total flow magnitude and magnitude of the flow in the fire regions
        flow_magnitude_in_fire = np.sum(magnitude[fire_mask > 0])
        total_flow_magnitude = np.sum(magnitude)

        if total_flow_magnitude == 0:
            return 0.0

        # normalize the total flow magnitude to a value between 0 and 1 based on a flow across the entire frame
        
        fire_prob = flow_magnitude_in_fire / total_flow_magnitude
        return fire_prob

        



