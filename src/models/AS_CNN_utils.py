import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_tree

from src.utils.joints import *


def adjacent_joints_distance(subject):
    """

    :param subject: shape (batch_size, 3, n_joints, seq_len)
    :return:
    """
    # skeleton shape (3, max_frame, n_joints = 25, n_subjects = 2)
    batch_size = subject.shape[0]
    max_frame = subject.shape[-1]
    n_neighbors = connexion_tuples.shape[0]

    distances_matrix = torch.zeros((batch_size, n_neighbors, max_frame))

    for j in range(n_neighbors):
        distances_matrix[:, j, :] = ((subject[:, 0, connexion_tuples[j, 0], :] - subject[:, 0, connexion_tuples[j, 1], :]) ** 2 + \
                                     (subject[:, 1, connexion_tuples[j, 0], :] - subject[:, 1, connexion_tuples[j, 1], :]) ** 2 + \
                                     (subject[:, 2, connexion_tuples[j, 0], :] - subject[:, 2, connexion_tuples[j, 1], :]) ** 2) ** (1 / 2)

    return distances_matrix


def compute_skeleton_graph(connexion_tuples):
    # connexion_tuples shape (n_neighboring_connections, )
    n_joints = np.amax(connexion_tuples) + 1

    skeleton_graph = np.zeros((n_joints, n_joints))

    for neighbor_joints in connexion_tuples:
        skeleton_graph[neighbor_joints[0], neighbor_joints[1]] = 1

    return skeleton_graph


def rescale_skeleton(subject, scale_factors):
    """

    :param subject: shape (batch_size, 3, n_joints, seq_len)
    :param scale_factors: shape (batch_size, n_bones)

    :return:
    """

    skeleton_graph_matrix = compute_skeleton_graph(connexion_tuples)
    skeleton_graph = csr_matrix(skeleton_graph_matrix)

    # Start at line 0
    queue = [0]

    while queue:
        # Get neighbors
        current_joint_idx = queue[0]
        neighbors = np.array(np.where(skeleton_graph_matrix[current_joint_idx, :] == 1)).transpose(1, 0)  # shape (n_neighbors, 1)

        for joint in neighbors:
            queue.append(int(joint))

        # Compute direction vector
        for i in range(neighbors.shape[0]):
            neighbor_idx = int(neighbors[i])

            # compute direction vector shape (batch_size, 3, max_frame)
            direction_vector = subject[:, :, neighbor_idx, :] - subject[:, :, current_joint_idx, :]

            # print(direction_vector[:, :, 0])

            # find index where connexion_tuple[index] = [current_joint_idx, neighbor_idx]
            connexion_tuple_idx = [i for i
                                   in range(connexion_tuples.shape[0])
                                   if np.array_equal(connexion_tuples[i, :],
                                                     np.array([current_joint_idx, neighbor_idx]))][0]

            # scale direction vector shape (batch_size, 3, max_frame)
            scaled_direction_vector = (scale_factors[:, connexion_tuple_idx] * direction_vector.permute(1, 2, 0)).permute(2, 0, 1)

            # print(scaled_direction_vector[:, :, 0])

            # compute translation_vector shape (batch_size, 3, max_frame)
            translation_vector = scaled_direction_vector - direction_vector

            # print(translation_vector[:, :, 0])

            # Propagate new position
            # 1. Find all children of neighboring joint
            children_graph_matrix = depth_first_tree(skeleton_graph, neighbor_idx, directed=True).toarray().astype(int)

            children = np.array(
                [i for i in range(children_graph_matrix.shape[0]) if np.sum(children_graph_matrix[:, i]) > 0])

            # 2. Propagate new position
            subject[:, :, neighbor_idx, :] += translation_vector
            for child in children: subject[:, :, child, :] += translation_vector

        queue.pop(0)

    return subject
