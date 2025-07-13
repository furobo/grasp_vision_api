# use opencv to calibrate eye-on-hand camera by aruco marker

import os
import cv2
import numpy as np
from cv2 import aruco
import transforms3d as tfs



def detect_aruco_marker(photo_dir, intrinsic=None, distCoeffs=np.zeros((5, 1)), marker_size=0.1):
    """Detects ArUco markers in an image and estimates their pose.

    Args:
        photo_dir (str): Directory containing the image file.
        intrinsic (np.ndarray, optional): Camera intrinsic matrix. Defaults to None.
        distCoeffs (np.ndarray, optional): Camera distortion coefficients. Defaults to np.zeros((5, 1)).
        marker_size (float, optional): Size of the marker in meters. Defaults to 0.1.

    Returns:
        np.ndarray: The rotation matrix of the detected marker if found, otherwise None.
    """
    gray_path = os.path.join(photo_dir, "rgb.png")
    print("pyhoo_dir", photo_dir)
    gray = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED)
    #aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) # debug
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    print("dict", aruco_dict)
    parameters =  aruco.DetectorParameters()
    print("para", parameters)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print("ids",ids)
    if ids is None:
        return None
    print("detect aruco marker numbers:", len(ids), corners)
    if len(ids) >= 1:
        rotation_vecs, translation_vecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix=intrinsic, distCoeffs=distCoeffs)
        mat, _ = cv2.Rodrigues(rotation_vecs[0][0])
        marker2camera_matrix = np.eye(4)
        marker2camera_matrix[:3,3] = translation_vecs[0][0]
        marker2camera_matrix[:3,:3] = mat
        print("mat", mat)
        print ("translation vector", translation_vecs[0][0])
        print ("rotation vector", rotation_vecs[0][0])
        translation_rotation_vecs = np.concatenate((translation_vecs[0][0], rotation_vecs[0][0]), axis=0)
        translation_rotation_path = os.path.join(photo_dir, "translation_RodriguesRotation.txt")
        np.savetxt(translation_rotation_path, translation_rotation_vecs, fmt='%.6f')
        return marker2camera_matrix
    else:
        return None


def calibrate_eye_on_hand(data_dir, num_photos, intrinsic, distCoeffs, marker_size):
    """Calibrates the eye-on-hand camera.

    Args:
        data_dir (str): Directory containing the calibration data.
        num_photos (int): Number of calibration photos.
        intrinsic (np.ndarray): Camera intrinsic matrix.
        distCoeffs (np.ndarray): Camera distortion coefficients.
        marker_size (float): Size of the ArUco marker in meters.

    Returns:
        np.ndarray: The transformation matrix from the camera to the TCP.
    """
    tcp2base_matrixes = []
    for i in range(num_photos):
        tcp2base_path = os.path.join(data_dir, "pose" + str(i), "pose.txt")
        tcp2base_pose = np.loadtxt(tcp2base_path) # mm, degree
        tcp2base_matrix = np.eye(4)
        tcp2base_matrix[:3, 3] = tcp2base_pose[:3]/1000 # convert mm to m
        euler_angles = [c*np.pi/180 for c in tcp2base_pose[3:]] # convert degree to radian
        tcp2base_matrix[:3, :3] = tfs.euler.euler2mat(*euler_angles)
        tcp2base_matrixes.append(tcp2base_matrix)

    marker2camera_matrixes = []
    for i in range(num_photos):
        photo_dir = os.path.join(data_dir, "pose" + str(i))
        marker2camera_matrix = detect_aruco_marker(photo_dir=photo_dir,
                                                   intrinsic=intrinsic,
                                                   distCoeffs=distCoeffs,
                                                   marker_size=marker_size)
        if marker2camera_matrix is not None:
            marker2camera_matrixes.append(marker2camera_matrix)
        else:
            print(f"===== No ArUco marker detected in photo {i}. Skipping...")
            return None

    if len(tcp2base_matrixes) != len(marker2camera_matrixes):
        raise ValueError("Mismatch between TCP-to-base and marker-to-camera matrices.")

    # Extract rotation and translation vectors for cv2.calibrateHandEye
    tcp2base_rotations = []
    tcp2base_translations = []
    marker2camera_rotations = []
    marker2camera_translations = []

    for tcp2base, marker2camera in zip(tcp2base_matrixes, marker2camera_matrixes):
        tcp2base_rotations.append(tcp2base[:3, :3])
        tcp2base_translations.append(tcp2base[:3, 3])
        marker2camera_rotations.append(marker2camera[:3, :3])
        marker2camera_translations.append(marker2camera[:3, 3])

    # Use OpenCV's calibrateHandEye to compute the eye-to-TCP transformation
    # method = cv2.CALIB_HAND_EYE_TSAI
    eye2tcp_rotation, eye2tcp_translation = cv2.calibrateHandEye(
        R_gripper2base=tcp2base_rotations,
        t_gripper2base=tcp2base_translations,
        R_target2cam=marker2camera_rotations,
        t_target2cam=marker2camera_translations,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    # Construct the eye-to-TCP transformation matrix
    eye2tcp_matrix = np.eye(4)
    eye2tcp_matrix[:3, :3] = eye2tcp_rotation
    eye2tcp_matrix[:3, 3] = eye2tcp_translation.squeeze()*1000 # Convert meters to mm

    print("Eye-to-TCP transformation matrix:")
    print(eye2tcp_matrix)

    # Save the transformation matrix to a file
    eye2tcp_path = os.path.join(data_dir, "eye2tcp_matrix.txt")
    np.savetxt(eye2tcp_path, eye2tcp_matrix, fmt="%.6f")
    print(f"Eye-to-TCP transformation matrix saved to {eye2tcp_path}")

    return eye2tcp_matrix

def evaluate(data_dir, num_photos):
    """Evaluates the calibration by calculating the mean and squared error of the marker's translation in the base frame.

    Args:
        data_dir (str): Directory containing the calibration data.
        num_photos (int): Number of calibration photos.

    Returns:
        dict: A dictionary containing the mean translation and squared error.
    """
    # Load the eye-to-TCP transformation matrix
    eye2tcp_path = os.path.join(data_dir, "eye2tcp_matrix.txt")
    if not os.path.exists(eye2tcp_path):
        raise FileNotFoundError(f"{eye2tcp_path} not found. Please run calibration first.")
    eye2tcp_matrix = np.loadtxt(eye2tcp_path)
    # convert to meters
    eye2tcp_matrix[:3, 3] = eye2tcp_matrix[:3, 3] / 1000
    

    tcp2base_matrixes = []
    for i in range(num_photos):
        tcp2base_path = os.path.join(data_dir, "pose" + str(i), "pose.txt")
        tcp2base_pose = np.loadtxt(tcp2base_path)  # mm, degree
        tcp2base_matrix = np.eye(4)
        tcp2base_matrix[:3, 3] = tcp2base_pose[:3] / 1000  # Convert mm to meters
        euler_angles = [c * np.pi / 180 for c in tcp2base_pose[3:]]  # Convert degrees to radians
        tcp2base_matrix[:3, :3] = tfs.euler.euler2mat(*euler_angles)
        tcp2base_matrixes.append(tcp2base_matrix)

    marker2camera_matrixes = []
    for i in range(num_photos):
        translation_rotation_path = os.path.join(data_dir, "pose" + str(i), "translation_RodriguesRotation.txt")
        if not os.path.exists(translation_rotation_path):
            raise FileNotFoundError(f"{translation_rotation_path} not found.")
        translation_rotation_vecs = np.loadtxt(translation_rotation_path)
        translation_vec = translation_rotation_vecs[:3]
        rotation_vec = translation_rotation_vecs[3:]
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        marker2camera_matrix = np.eye(4)
        marker2camera_matrix[:3, 3] = translation_vec
        marker2camera_matrix[:3, :3] = rotation_matrix
        marker2camera_matrixes.append(marker2camera_matrix)

    if len(tcp2base_matrixes) != len(marker2camera_matrixes):
        raise ValueError("Mismatch between TCP-to-base and marker-to-camera matrices.")

    # Calculate marker translation in the base frame
    marker_translations_in_base = []
    for tcp2base, marker2camera in zip(tcp2base_matrixes, marker2camera_matrixes):
        marker_in_base = tcp2base @ eye2tcp_matrix @ marker2camera
        marker_translations_in_base.append(marker_in_base[:3, 3])

    # Compute mean, RMSE, and MAE of translations
    marker_translations_in_base = np.array(marker_translations_in_base)
    mean_translation = np.mean(marker_translations_in_base, axis=0)

    # Calculate individual errors for each pose
    individual_errors = np.linalg.norm(marker_translations_in_base - mean_translation, axis=1)

    # Calculate RMSE and MAE
    rmse = np.sqrt(np.mean(individual_errors ** 2))  # Root Mean Squared Error
    mae = np.mean(individual_errors)  # Mean Absolute Error

    print("Mean translation of the marker in the base frame:", mean_translation)
    for i in range(num_photos):
        print(f"Error for pose {i}: {individual_errors[i]}")

    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)

    result = {
        "mean_translation": mean_translation,
        "rmse": rmse,
        "mae": mae,
        "individual_errors": individual_errors  
    }

    return result


def start_calibration(intrinsic_path = "data/intrinsic.txt", num_photos = 7):
    print("Calibration Eye-on-hand Camera")
    
    if os.path.exists(intrinsic_path):
        intrinsic = np.loadtxt(intrinsic_path)
    # else:
        
    #     from orbbec_sdk_interface import OrbbecSDKInterface
    #     cam = OrbbecSDKInterface()
    #     intrinsic = cam.get_camera_intrinsic()  # read as a numpy array
    #     print("intrinsic matrix:", intrinsic)  # save to a txt file with precision as 6
    #     np.savetxt("data/intrinsic.txt", intrinsic, fmt='%.6f')


    
    calibrate_eye_on_hand(data_dir="data",
                        num_photos = 7,
                        intrinsic=intrinsic,
                        distCoeffs=np.zeros((5, 1)),
                        marker_size=0.1)
    result = evaluate(data_dir="data", num_photos=num_photos)
    return result


if __name__ == "__main__":
    result = start_calibration()
    print("Calibration done")
    print(result)