import numpy as np

from collections import deque # ordererd collection with ends

from skimage import transform # for preprocessing frames
from skimage.color import rgb2gray # convert frames to grayscale

from config import *

# To reduce the computing time, we reduce the frame size and dimensionality and normalize
def preprocess_frame(frame):
    # gray scale frame
    frame_gray = rgb2gray(frame)
    
    # crop the screen (no hud)
    frame_cropped = frame_gray[40:, :]
    
    # normalize pixel values
    frame_normalized = frame_cropped / 255.0
    
    # downscale frame
    frame_preprocessed = transform.resize(frame_cropped, [100, 128])
    
    return frame_preprocessed

# stack frames to get a sense of motion
def stack_frames(stacked_frames, state, is_new_episode):
    # preprocess frames
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # clear our stacked frames
        stacked_frames = deque([np.zeros((100, 128), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # new episode => copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # append frame to deque
        stacked_frames.append(frame)
        
        # build the stacked state
        stacked_state = np.stack(stacked_frames, axis=2)
        
    return stacked_state, stacked_frames