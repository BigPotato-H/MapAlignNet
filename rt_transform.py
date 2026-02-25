import torch

def euler_to_rot_matrix_batch(euler_angles):
    """
    Convert a batch of Euler angles (XYZ order) to a rotation matrix.

    Parameters:
    euler_angles (torch.Tensor): Tensor of shape (B, 3), where B is the batch size.

    Returns:
    torch.Tensor: Tensor of shape (B, 3, 3) containing rotation matrices for each batch element.
    """
    theta_x, theta_y, theta_z = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    cos_x = torch.cos(theta_x)
    sin_x = torch.sin(theta_x)
    cos_y = torch.cos(theta_y)
    sin_y = torch.sin(theta_y)
    cos_z = torch.cos(theta_z)
    sin_z = torch.sin(theta_z)

    # Rotation matrix for X axis
    rot_x = torch.stack([torch.ones_like(cos_x), torch.zeros_like(cos_x), torch.zeros_like(cos_x),
                         torch.zeros_like(cos_x), cos_x, -sin_x,
                         torch.zeros_like(cos_x), sin_x, cos_x], dim=-1).view(-1, 3, 3)

    # Rotation matrix for Y axis
    rot_y = torch.stack([cos_y, torch.zeros_like(cos_y), sin_y,
                         torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y),
                         -sin_y, torch.zeros_like(cos_y), cos_y], dim=-1).view(-1, 3, 3)

    # Rotation matrix for Z axis
    rot_z = torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z),
                         sin_z, cos_z, torch.zeros_like(cos_z),
                         torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)], dim=-1).view(-1, 3, 3)

    # Combined rotation matrix R = Rz * Ry * Rx (XYZ rotation convention)
    rotation_matrix = torch.matmul(rot_z, torch.matmul(rot_y, rot_x))

    return rotation_matrix


def output_to_transform_matrix(network_output):
    """
    Convert the output of a network (B, 6) to a 3x4 transformation matrix.

    Parameters:
    network_output (torch.Tensor): Tensor of shape (B, 6), where B is the batch size.
                                   First 3 elements are translation, next 3 are Euler angles (radians).

    Returns:
    torch.Tensor: Tensor of shape (B, 3, 4) representing the 3x4 transformation matrices.
    """
    # Split the network output into translation and euler angles
    #print(network_output[0].detach().cpu().numpy())
    translation = network_output[:, :3]  # Shape: (B, 3)
    euler_angles = network_output[:, 3:]  # Shape: (B, 3)

    # Convert Euler angles to rotation matrix (B, 3, 3)
    rotation_matrix = euler_to_rot_matrix_batch(euler_angles)

    # Combine rotation matrix and translation to form a 3x4 transformation matrix
    # Initialize a tensor to hold the transformation matrix
    B = network_output.shape[0]  # Batch size
    transform_matrix = torch.zeros((B, 3, 4), dtype=network_output.dtype, device=network_output.device)

    # Set the rotation part (first 3x3 block)
    transform_matrix[:, :, :3] = rotation_matrix

    # Set the translation part (last column)
    transform_matrix[:, :, 3] = translation

    return transform_matrix


import torch


def rot_matrix_to_euler_batch(rotation_matrix):
    """
    Convert a batch of rotation matrices (B, 3, 3) to Euler angles (B, 3).
    Assumes XYZ rotation convention (extrinsic rotations).

    Parameters:
    rotation_matrix (torch.Tensor): Rotation matrices of shape (B, 3, 3).

    Returns:
    torch.Tensor: Euler angles in radians of shape (B, 3) for each batch.
    """
    sy = torch.sqrt(rotation_matrix[:, 0, 0] ** 2 + rotation_matrix[:, 1, 0] ** 2)

    singular = sy < 1e-6  # Check if the matrix is close to a singular case

    # Initialize the tensor for Euler angles
    euler_angles = torch.zeros(rotation_matrix.shape[0], 3, device=rotation_matrix.device)

    if not singular.any():
        euler_angles[:, 0] = torch.atan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2])  # theta_x
        euler_angles[:, 1] = torch.atan2(-rotation_matrix[:, 2, 0], sy)  # theta_y
        euler_angles[:, 2] = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])  # theta_z
    else:
        euler_angles[:, 0] = torch.atan2(-rotation_matrix[:, 1, 2], rotation_matrix[:, 1, 1])  # theta_x
        euler_angles[:, 1] = torch.atan2(-rotation_matrix[:, 2, 0], sy)  # theta_y
        euler_angles[:, 2] = 0  # theta_z is set to 0 for singular cases

    return euler_angles


def transform_matrix_to_euler_and_translation(transform_matrix):
    """
    Convert a batch of transformation matrices (B, 3, 4) to (B, 6) where the
    first 3 elements are translation and the next 3 are Euler angles in radians.

    Parameters:
    transform_matrix (torch.Tensor): Transformation matrices of shape (B, 3, 4).

    Returns:
    torch.Tensor: Tensor of shape (B, 6), with the first 3 elements being the translation
                  and the next 3 being the Euler angles in radians.
    """
    # Extract the rotation matrix (first 3x3 block)
    rotation_matrix = transform_matrix[:, :, :3]  # Shape: (B, 3, 3)

    # Extract the translation vector (last column)
    translation = transform_matrix[:, :, 3]  # Shape: (B, 3)

    # Convert rotation matrix to Euler angles
    euler_angles = rot_matrix_to_euler_batch(rotation_matrix)

    # Concatenate the translation and Euler angles to form a (B, 6) tensor
    transform_params = torch.cat([translation, euler_angles], dim=1)  # Shape: (B, 6)

    return transform_params


def transformation_loss(predicted, GT):
    """
    Calculate the loss between predicted transformation matrices and ground truth (GT) matrices.

    Parameters:
    predicted (torch.Tensor): Predicted transformation matrices of shape (B, 3, 4).
    GT (torch.Tensor): Ground truth transformation matrices of shape (B, 3, 4).

    Returns:
    torch.Tensor: Combined loss value (rotation loss + translation loss).
    """
    # Split the predicted and ground truth into rotation (3x3) and translation (3x1) parts
    pred_rotation = predicted[:, :, :3]  # Shape: (B, 3, 3)
    pred_translation = predicted[:, :, 3]  # Shape: (B, 3)

    GT_rotation = GT[:, :, :3]  # Shape: (B, 3, 3)
    GT_translation = GT[:, :, 3]  # Shape: (B, 3)

    # Rotation Loss: Frobenius norm (L2 norm) of the difference between the rotation matrices
    rotation_loss = torch.norm(pred_rotation - GT_rotation, p='fro', dim=(1, 2))  # Frobenius norm for each batch

    # Translation Loss: Euclidean distance between the translation vectors
    translation_loss = torch.norm(pred_translation - GT_translation, p=2, dim=1)  # Euclidean norm for each batch

    # Combine the two losses (rotation and translation)
    combined_loss = rotation_loss + translation_loss  # You can apply weights if needed

    # Return the mean loss over the batch
    return torch.relu(combined_loss.mean())

def transformation_uncertainty_loss(predicted, GT,uncertainty):
    """
    Calculate the loss between predicted transformation matrices and ground truth (GT) matrices.

    Parameters:
    predicted (torch.Tensor): Predicted transformation matrices of shape (B, 3, 4).
    GT (torch.Tensor): Ground truth transformation matrices of shape (B, 3, 4).

    Returns:
    torch.Tensor: Combined loss value (rotation loss + translation loss).
    """
    # Split the predicted and ground truth into rotation (3x3) and translation (3x1) parts
    pred_rotation = predicted[:, :, :3]  # Shape: (B, 3, 3)
    pred_translation = predicted[:, :, 3]  # Shape: (B, 3)

    GT_rotation = GT[:, :, :3]  # Shape: (B, 3, 3)
    GT_translation = GT[:, :, 3]  # Shape: (B, 3)

    # Rotation Loss: Frobenius norm (L2 norm) of the difference between the rotation matrices
    rotation_loss = torch.norm(pred_rotation - GT_rotation, p='fro', dim=(1, 2))  # Frobenius norm for each batch

    # Translation Loss: Euclidean distance between the translation vectors
    translation_loss = torch.norm(pred_translation - GT_translation, p=2, dim=1)  # Euclidean norm for each batch

    # Extract uncertainties and ensure non-negativity using exponential
    rotation_uncertainty = torch.exp(uncertainty[:, :3])
    translation_uncertainty = torch.exp(uncertainty[:, 3:])

    # Weighted losses using uncertainties
    rotation_loss_weighted = rotation_loss / rotation_uncertainty + torch.log(rotation_uncertainty)
    translation_loss_weighted = translation_loss / translation_uncertainty + torch.log(translation_uncertainty)

    # Combine the two losses (rotation and translation)
    combined_loss = rotation_loss_weighted  + translation_loss_weighted   # You can apply weights if needed

    # Return the mean loss over the batch
    return combined_loss.mean()


def test(transform_matrix, tolerance=1e-6):
    """
    Test the conversion from transformation matrix -> Euler angles + translation and back.

    Parameters:
    transform_matrix (torch.Tensor): Input transformation matrix of shape (B, 3, 4).
    tolerance (float): Tolerance for comparing the original and reconstructed matrices.

    Returns:
    bool: True if the test passes, False otherwise.
    """
    # Step 1: Convert transformation matrix to translation + Euler angles
    params = transform_matrix_to_euler_and_translation(transform_matrix)

    # Extract translation and Euler angles from params (B, 6)
    translation = params[:, :3]
    euler_angles = params[:, 3:]

    # Step 2: Convert back to transformation matrix
    # Convert Euler angles to rotation matrices
    rotation_matrix = euler_to_rot_matrix_batch(euler_angles)

    # Combine rotation matrix and translation into a 3x4 transformation matrix
    B = transform_matrix.shape[0]  # Batch size
    reconstructed_transform_matrix = torch.zeros((B, 3, 4), dtype=transform_matrix.dtype,
                                                 device=transform_matrix.device)
    reconstructed_transform_matrix[:, :, :3] = rotation_matrix
    reconstructed_transform_matrix[:, :, 3] = translation

    # Step 3: Compare original transformation matrix with reconstructed transformation matrix
    difference = torch.norm(transform_matrix - reconstructed_transform_matrix, p='fro', dim=(1, 2))

    # Check if the maximum difference is within the specified tolerance
    if torch.all(difference < tolerance):
        print("Test passed! The transformation conversion is consistent.")
        return True
    else:
        print("Test failed! The transformation conversion is not consistent.")
        return False

