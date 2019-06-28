# Hyper parameters
crop_size = 50

import numpy as np
from joints import *


def extract_hands(skeleton_rgb, videodata, crop_size):
    # videodata shape (max_frame, 1080, 1920, 3)
    # skeleton_rgb shape (2{yx}, max_frame, num_joint=25, 2)
    hand_crops = []
    
    max_x = videodata.shape[1]
    max_y = videodata.shape[2]
    
    offset = int(crop_size / 2)
    
    n_frames = skeleton_rgb.shape[1]
    n_subjects = 1
    
    # Check if some coordinates for skeleton 2 are != 0
    if np.any(skeleton_rgb[:, :, :, 1]):
        n_subjects = 2
    
    # Get correspoding time coordinates
    for s in range(n_subjects):
        hand_crops_s = np.zeros((n_frames, 2, crop_size, crop_size, 3), dtype=np.uint8)
        
        for t in range(n_frames):
            # Get right/left hand center coordinates            
            left_hand_x = max(min(int(np.nan_to_num(skeleton_rgb[1, t, Joints.HANDLEFT, s])), max_x), 0)
            left_hand_y = max(min(int(np.nan_to_num(skeleton_rgb[0, t, Joints.HANDLEFT, s])), max_y), 0)
            
            right_hand_x = max(min(int(np.nan_to_num(skeleton_rgb[1, t, Joints.HANDRIGHT, s])), max_x), 0)
            right_hand_y = max(min(int(np.nan_to_num(skeleton_rgb[0, t, Joints.HANDRIGHT, s])), max_y), 0)
            
            frame = np.pad(videodata[t], ((offset, offset), (offset, offset), (0, 0)), mode='constant') # shape(1130, 1970, 3)
            
            hand_crops_s[t, 0] = frame[left_hand_x:left_hand_x+2*offset, left_hand_y:left_hand_y+2*offset]
            hand_crops_s[t, 1] = frame[right_hand_x:right_hand_x+2*offset, right_hand_y:right_hand_y+2*offset]
            
        hand_crops.append(hand_crops_s)
    
    return np.concatenate(hand_crops, axis = 1)





# Code courtesy of yysijie and the awesome paper ST-GCN
# https://github.com/yysijie/st-gcn/
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    data = np.around(data, decimals=3)
    return data

def read_color_xy(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((2, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['colorX'], v['colorY']]
                else:
                    pass
    return data