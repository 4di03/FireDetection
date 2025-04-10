"""
Adithya Palle
Mar 31, 2025
Final Project

File containing functionality for computing optical flow between two frames and using it to predict fire probability.
"""
from video_model import VideoModel
import cv2
import numpy as np
import torch
DEBUG = False
FIRE_LOWER_RGB = np.array([150,  50,   0])   # R, G, B
FIRE_UPPER_RGB = np.array([255, 255, 150])
def draw_flow_on_white(flow, shape, step=16):
    """
    Draws optical flow vectors as arrows on a white background.

    Args:
        flow (np.ndarray): Optical flow of shape (H, W, 2).
        shape (tuple): Height and width of the frame.
        step (int): Sampling stride for arrow spacing.

    Returns:
        np.ndarray: White image with flow arrows.
    """
    h, w = shape
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
    fx, fy = flow[y, x].T

    for (x1, y1, dx, dy) in zip(x.flatten(), y.flatten(), fx.flatten(), fy.flatten()):
        x2, y2 = int(x1 + dx), int(y1 + dy)
        cv2.arrowedLine(white_bg, (x1, y1), (x2, y2), (0, 0, 0), 1, tipLength=0.3)

    return white_bg
class FarnebackOpticalFlowModel(VideoModel):

    def predict_on_last_frame(self, frames : list[torch.Tensor]) -> float:
        """
        Predicts the fire probability for the last frame in the given stream of images using Farneback Optical Flow.
        This takes the last two frames and computes the optical flow between them.
        It then determines the flow vectors for color regions that match fire (orange-red)
        produces a fire probability between 0 and 1 based on the sumemd flow in those regions.
        Args:
            frames: A list of tensors representing the frames in the video stream. The last frame is the most recent one for which the prediction is made.
        Returns:
            A float representing the fire probability for the last frame in the video stream.
            This can be thought of as the % of total motion in the image that is made up of fire-like motion
        """
        if len(frames) < 2:
            return 0.0
        
        prev_frame_np = frames[-2].permute(1,2,0).numpy()
        target_frame_np = frames[-1].permute(1,2,0).numpy()

        # # print max and min values of the frames
        # print(f"prev_frame_np max: {np.max(prev_frame_np)}, min: {np.min(prev_frame_np)}")
        # print(f"target_frame_np max: {np.max(target_frame_np)}, min: {np.min(target_frame_np)}")
        
        # Convert the last two frames to grayscale, and convert (C, H, W) to (H, W, C)
        prev_frame = cv2.cvtColor(prev_frame_np, cv2.COLOR_RGB2GRAY)
        pred_frame = cv2.cvtColor(target_frame_np, cv2.COLOR_RGB2GRAY)

        # Compute the optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, pred_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Calculate the magnitude of the flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Create a mask for the fire color range (orange-red)
        # we mask previous frame since flow[i,j] = (dx, dy) represents the flow from the previous frame to the current frame, so we want to observe flow in pixels where there was previously fire-like color
        fire_mask = cv2.inRange(prev_frame_np, FIRE_LOWER_RGB, FIRE_UPPER_RGB)
        # Calculate the sum of the flow magnitudes in the fire regions
        flow_magnitude_in_fire = np.sum(magnitude[fire_mask > 0])
        # Calculate the total flow magnitude and magnitude of the flow in the fire regions
        flow_magnitude_in_fire = np.sum(magnitude[fire_mask > 0])
        total_flow_magnitude = np.sum(magnitude)

        if total_flow_magnitude == 0:
            return 0.0

        # normalize the total flow magnitude to a value between 0 and 1 based on a flow across the entire frame
        # this can be thought of as the % of total motion that is made up of fire-like motion
        fire_prob = flow_magnitude_in_fire / total_flow_magnitude

        if DEBUG:
            # show the flow field and the original frame and the fire mask
            cv2.imshow("Flow Field", draw_flow_on_white(flow, target_frame_np.shape[:2]))
            cv2.imshow("Original Frame", cv2.cvtColor(target_frame_np, cv2.COLOR_BGR2RGB))
            cv2.imshow("Fire Mask", fire_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        return fire_prob

        



