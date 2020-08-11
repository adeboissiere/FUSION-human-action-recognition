r"""
Contains functions to augment skeleton data.
Skeleton data has a prior normalization step applied where the scene is translated from the camera coordinate system
to a new local coordinate system. The translation is given by the vector formed between the origin of the camera and
the first subject's SPINE_MID joint for the first frame. The vector is the same for all subsequent frames.

The skeleton data used as inputs is already "prior" normalized.

Three functions are provided.

    - *build_rotation_matrix*: Creates a 3x3 rotation matrix for a given axis
    - *rotate_skeleton*: Randomly rotates a skeleton sequence around the X, Y and Z axis.
    - *stretched_image_from_skeleton_sequence*: Creates an RGB image from a skeleton sequence

**Note** that the rotation angles are randomly taken in between hardcoded global variables in this file.

As more guidelines we add the following informations.

Kinect v2 coordinate system:
    - **x** : horizontal plane
    - **y** : height
    - **z** : depth

NTU RGB-D sequences are acquired from -45° to 45° on the x axis

"""

import math
import cv2
from src.utils.joints import *


# Global variables
x_rot_low = -20.0
x_rot_high = 20.0
y_rot_low = -20.0
y_rot_high = 20.0
z_rot_low = -20.0
z_rot_high = 20.0


def build_rotation_matrix(axis, rot_angle):
    r"""Builds a random rotation matrix for a given axis.

    Inputs:
        - **axis** (int): Axis of rotation (0: x, 1: y, 2: z)
        - **rot_angle** (float): Angle of rotation in degrees

    Outputs:
        **rotation_matrix (np array)**: 3x3 rotation matrix

    """

    rotation_matrix = np.zeros((3, 3))

    cos_rotation_angle = np.cos(rot_angle)
    sin_rotation_angle = np.sin(rot_angle)

    if axis == 0:
        rotation_matrix[0, 0] = 1
        rotation_matrix[1, 1] = cos_rotation_angle
        rotation_matrix[1, 2] = - sin_rotation_angle
        rotation_matrix[2, 1] = sin_rotation_angle
        rotation_matrix[2, 2] = cos_rotation_angle
    elif axis == 1:
        rotation_matrix[1, 1] = 1
        rotation_matrix[0, 0] = cos_rotation_angle
        rotation_matrix[2, 0] = - sin_rotation_angle
        rotation_matrix[0, 2] = sin_rotation_angle
        rotation_matrix[2, 2] = cos_rotation_angle

    elif axis == 2:
        rotation_matrix[2, 2] = 1
        rotation_matrix[0, 0] = cos_rotation_angle
        rotation_matrix[0, 1] = - sin_rotation_angle
        rotation_matrix[1, 0] = sin_rotation_angle
        rotation_matrix[1, 1] = cos_rotation_angle

    return rotation_matrix


def get_mirror_skeleton(skeleton):
    r"""Mirrors the skeleton. Eg. left hand becomes right hand, etc.

    Inputs:
        **skeleton** (np array): Skeleton sequence of shape `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`

    Outputs:
        **skeleton** (np array): Randomly rotated skeleton sequence of shape
        `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`
    """

    skeleton[:, :, Joints.SHOULDERLEFT, :], skeleton[:, :, Joints.SHOULDERRIGHT] = \
        skeleton[:, :, Joints.SHOULDERRIGHT].copy(), skeleton[:, :, Joints.SHOULDERLEFT, :].copy()
    skeleton[:, :, Joints.ELBOWLEFT, :], skeleton[:, :, Joints.ELBOWRIGHT] = \
        skeleton[:, :, Joints.ELBOWRIGHT].copy(), skeleton[:, :, Joints.ELBOWLEFT, :].copy()
    skeleton[:, :, Joints.WRISTLEFT, :], skeleton[:, :, Joints.WRISTRIGHT] = \
        skeleton[:, :, Joints.WRISTRIGHT].copy(), skeleton[:, :, Joints.WRISTLEFT, :].copy()
    skeleton[:, :, Joints.HANDLEFT, :], skeleton[:, :, Joints.HANDRIGHT] = \
        skeleton[:, :, Joints.HANDRIGHT].copy(), skeleton[:, :, Joints.HANDLEFT, :].copy()
    skeleton[:, :, Joints.HIPLEFT, :], skeleton[:, :, Joints.HIPRIGHT] = \
        skeleton[:, :, Joints.HIPRIGHT].copy(), skeleton[:, :, Joints.HIPLEFT, :].copy()
    skeleton[:, :, Joints.KNEELEFT, :], skeleton[:, :, Joints.KNEERIGHT] = \
        skeleton[:, :, Joints.KNEERIGHT].copy(), skeleton[:, :, Joints.KNEELEFT, :].copy()
    skeleton[:, :, Joints.ANKLELEFT, :], skeleton[:, :, Joints.ANKLERIGHT] = \
        skeleton[:, :, Joints.ANKLERIGHT].copy(), skeleton[:, :, Joints.ANKLELEFT, :].copy()
    skeleton[:, :, Joints.FOOTLEFT, :], skeleton[:, :, Joints.FOOTRIGHT] = \
        skeleton[:, :, Joints.FOOTRIGHT].copy(), skeleton[:, :, Joints.FOOTLEFT, :].copy()
    skeleton[:, :, Joints.HANDTIPLEFT, :], skeleton[:, :, Joints.HANDTIPRIGHT] = \
        skeleton[:, :, Joints.HANDTIPRIGHT].copy(), skeleton[:, :, Joints.HANDTIPLEFT, :].copy()
    skeleton[:, :, Joints.THUMBLEFT, :], skeleton[:, :, Joints.THUMBRIGHT] = \
        skeleton[:, :, Joints.THUMBRIGHT].copy(), skeleton[:, :, Joints.THUMBLEFT, :].copy()

    return skeleton


def rotate_skeleton(skeleton):
    r"""Rotates the skeleton sequence around its different axis.

    Inputs:
        **skeleton** (np array): Skeleton sequence of shape `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`

    Outputs:
        **skeleton_aug** (np array): Randomly rotated skeleton sequence of shape
        `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`

    """

    seq_len = skeleton.shape[1]
    n_joints = skeleton.shape[2]
    n_subjects = skeleton.shape[3]

    # Define rotation angles
    x_rot = np.random.uniform(x_rot_low, x_rot_high)
    y_rot = np.random.uniform(y_rot_low, y_rot_high)
    z_rot = np.random.uniform(z_rot_low, z_rot_high)

    # Transform degrees to radians
    x_rot = x_rot * math.pi / 180
    y_rot = y_rot * math.pi / 180
    z_rot = z_rot * math.pi / 180

    # Compute rotation matrix
    rotation_matrix_x = build_rotation_matrix(0, x_rot)
    rotation_matrix_y = build_rotation_matrix(1, y_rot)
    rotation_matrix_z = build_rotation_matrix(2, z_rot)
    rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    # Reshape skeleton
    # -> shape (3, seq_len, n_joints * n_subjects)
    # Ordered in such way that skeleton[:, :, subject_id*n_joints:(subject_id+1)*n_joints gives the coordinates
    # of subject number "subject_id"
    skeleton = np.reshape(skeleton, (3, seq_len, n_joints * n_subjects), order='F')

    # Apply rotation matrix
    skeleton_aug = skeleton.transpose(1, 2, 0) @ rotation_matrix.transpose(1, 0)[None, ...]

    # Transpose back
    skeleton_aug = skeleton_aug.transpose(2, 0, 1) # shape (3, seq_len, n_joints * n_subjects)
    skeleton_aug = np.reshape(skeleton_aug, (3, seq_len, n_joints, n_subjects), order='F')
    # Final shape (3, max_frame, num_joint=25, 2) : same as original

    return skeleton_aug


def stretched_image_from_skeleton_sequence(skeleton, c_min, c_max):
    r"""Rotates the skeleton sequence around its different axis.

    Inputs:
        - **skeleton** (np array): Skeleton sequence of shape `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`
        - **c_min** (int): Minimum coordinate value across all sequences, joints, subjects, frames after the prior
          normalization step.
        - **c_max** (int): Maximum coordinate value across all sequences, joints, subjects, frames after the prior
          normalization step.

    Outputs:
        **skeleton_image** (np array): RGB image of shape `(3, 224, 224)`

    """

    max_frame = skeleton.shape[1]
    n_joints = skeleton.shape[2]

    # Reshape skeleton coordinates into an image
    skeleton_image = np.zeros((3, max_frame, 2 * n_joints))
    skeleton_image[:, :, 0:n_joints] = skeleton[:, :, :, 0]
    skeleton_image[:, :, n_joints:2 * n_joints] = skeleton[:, :, :, 1]
    skeleton_image = np.transpose(skeleton_image, (0, 2, 1))

    # Normalize (min-max)
    skeleton_image = np.floor(255 * (skeleton_image - c_min) / (c_max - c_min))  # shape (3, 2 * n_joints, max_frame)

    # Reshape image for ResNet
    skeleton_image = cv2.resize(skeleton_image.transpose(1, 2, 0), dsize=(224, 224)).transpose(2, 0, 1)

    return skeleton_image

