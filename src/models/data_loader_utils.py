'''
Kinect coordinate system :
x : horizontal plane
y : height
z : depth

NTU RGB-D acquisition are rotated from -45° to 45° on the x axis

Try :
x_rot € [-45°, 45°]
y_rot € [-90°, 90°]
z_rot € [-10°, 10°]
'''

import numpy as np
import math
import cv2
from src.utils.joints import *

# Global variables
x_rot_low = -17.0
x_rot_high = 17.0
y_rot_low = -17.0
y_rot_high = 17.0
z_rot_low = -17.0
z_rot_high = 17.0

# Global variables
'''
x_rot_low = 0
x_rot_high = 0
y_rot_low = 0
y_rot_high = 0
z_rot_low = 0
z_rot_high = 0
'''


def build_rotation_matrix(axis, rot_angle):
    '''
    Builds a rotation matrix for a given axis.

    :param axis: Axis of rotation (0: x, 1: y, 2: z)
    :param rot_angle: Angle of rotation in degrees

    :return rotation_matrix: Rotation matrix (https://fr.wikipedia.org/wiki/Matrice_de_rotation)
    '''
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


def rotate_skeleton(skeleton):
    '''
    Rotates the skeleton around its different axis

    :param skeleton: np array of shape (3, max_frame, num_joint=25, 2). The first frame should be normalized to the
        position of the torso of the first subject

    :return skeleton_aug: augmented skeleton
    '''

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
    assert np.array_equal(skeleton[:, :, :, 0], skeleton.reshape((3, seq_len, n_joints * n_subjects), order='F')[:, :,
                                   0 * n_joints:n_joints])
    assert np.array_equal(skeleton[:, :, :, 1], skeleton.reshape((3, seq_len, n_joints * n_subjects), order='F')[:, :,
                                                1 * n_joints:2*n_joints])

    skeleton = np.reshape(skeleton, (3, seq_len, n_joints * n_subjects), order='F')

    # print(skeleton[0, 10, :])

    # Apply rotation matrix
    skeleton_aug = skeleton.transpose(1, 2, 0) @ rotation_matrix.transpose(1, 0)[None, ...]
    # print(skeleton.transpose(1, 2, 0).shape)
    # print(rotation_matrix.transpose(1, 0)[None, ...].shape)

    # Transpose back
    skeleton_aug = skeleton_aug.transpose(2, 0, 1) # shape (3, seq_len, n_joints * n_subjects)
    skeleton_aug = np.reshape(skeleton_aug, (3, seq_len, n_joints, n_subjects), order='F')
    # Final shape (3, max_frame, num_joint=25, 2) : same as original

    return skeleton_aug


def create_stretched_image_from_skeleton_sequence(skeleton, c_min, c_max):
    max_frame = skeleton.shape[1]
    n_joints = skeleton.shape[2]

    # Reshape skeleton coordinates into an image
    skeleton_image = np.zeros((3, max_frame, 2 * n_joints))
    skeleton_image[:, :, 0:n_joints] = skeleton[:, :, :, 0]
    skeleton_image[:, :, n_joints:2 * n_joints] = skeleton[:, :, :, 1]
    skeleton_image = np.transpose(skeleton_image, (0, 2, 1))

    # Normalize
    skeleton_image = np.floor(255 * (skeleton_image - c_min) / (c_max - c_min))  # shape (3, 2 * n_joints, max_frame)

    # Reshape image for ResNet
    skeleton_image = cv2.resize(skeleton_image.transpose(1, 2, 0), dsize=(224, 224)).transpose(2, 0, 1)

    return skeleton_image


def create_padded_image_from_skeleton_sequence(skeleton, c_min, c_max):
    max_frame = skeleton.shape[1]
    n_joints = skeleton.shape[2]

    # Reshape skeleton coordinates into an image
    skeleton_image = np.zeros((3, max_frame, 2 * n_joints))
    skeleton_image[:, :, 0:n_joints] = skeleton[:, :, :, 0]
    skeleton_image[:, :, n_joints:2 * n_joints] = skeleton[:, :, :, 1]
    skeleton_image = np.transpose(skeleton_image, (0, 2, 1))

    # Normalize
    skeleton_image = np.floor(255 * (skeleton_image - c_min) / (c_max - c_min))  # shape (3, 2 * n_joints, max_frame)
    skeleton_image = np.concatenate([skeleton_image, np.zeros((3, 224 - 2 * n_joints, max_frame))], axis=1)

    # Reshape image for ResNet
    skeleton_image = cv2.resize(skeleton_image.transpose(1, 2, 0), dsize=(224, 224)).transpose(2, 0, 1)

    return skeleton_image


def compute_avg_bone_length(skeleton):
    # skeleton shape (3, max_frame, n_joints = 25, n_subjects = 2)
    max_frame = skeleton.shape[1]
    n_neighbors = connexion_tuples.shape[0]
    n_subjects = skeleton.shape[-1]

    distances_matrix = np.zeros((n_neighbors, max_frame, 2))

    for j in range(n_neighbors):
        distances_matrix[j, :, :] = ((skeleton[0, :, connexion_tuples[j, 0], :] - skeleton[0, :, connexion_tuples[j, 1], :]) ** 2 + \
                                     (skeleton[1, :, connexion_tuples[j, 0], :] - skeleton[1, :, connexion_tuples[j, 1], :]) ** 2 + \
                                     (skeleton[2, :, connexion_tuples[j, 0], :] - skeleton[2, :, connexion_tuples[j, 1], :]) ** 2) ** (1 / 2)

    distances_matrix = np.mean(distances_matrix, axis=1)
    distances_matrix = distances_matrix.reshape(n_neighbors * n_subjects)

    return distances_matrix
