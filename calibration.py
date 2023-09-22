import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

square_size = 24  # the size of marker
image_path = "images/"  # Path where the calibration images are stored
pattern_size = (10, 7)  # Size of the chessboard grid
obj_points_3D = []
img_points_2D = []

# Defining the criteria for corner refinement in the findChessboardCorners function
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibration(image_path, obj_point, pattern_size, criteria):
    image_shape = None  # Will store the shape of the image
    obj_3d = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj_3d[:, :2] = np.mgrid[0: pattern_size[0], 0: pattern_size[1]].T.reshape(-1, 2)
    obj_3d *= square_size  # Scaling the object points by the actual size of the squares

    # Checking if there is enough data for calibration
    if len(obj_points_3D) > 0 and len(img_points_2D) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_3D, img_points_2D, image_shape[::-1], None, None)
    else:
        print(f"Insufficient data: {len(obj_points_3D)} object points and {len(img_points_2D)} image points")

    # Iterating through each image in the directory
    for file in os.listdir(image_path):
        imagePath = os.path.join(image_path, file)
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Failed to load image: {file}")
            continue

        image_shape = image.shape
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Finding the chessboard corners
        ret, corners = cv2.findChessboardCorners(image, pattern_size, None)

        if ret:  # If corners are found
            print('Shape of corners:', corners.shape)
            obj_points_3D.append(obj_3d)  # Append 3D points
            # Refining the corner locations
            corners2 = cv2.cornerSubPix(gray_img, corners, (3, 3), (-1, -1), criteria)
            img_points_2D.append(corners2)  # Append 2D points

            # Draw corners on the image for visualization
            img = cv2.drawChessboardCorners(image, pattern_size, corners2, ret)
            cv2.imshow('img1', img)
            cv2.waitKey(500)  # Wait for 500 ms
    if image_shape is None:
        print('No images were loaded successfully')
        return None

    cv2.destroyAllWindows()

    if len(obj_points_3D) > 0 and len(img_points_2D) > 0:
        # Performing the camera calibration again
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_3D, img_points_2D, gray_img.shape[::-1], None,
                                                           None)
        print('Calibration Done')
        return mtx, dist, rvecs, tvecs, obj_points_3D, img_points_2D
    else:
        print(f"Insufficient data: {len(obj_points_3D)} object points and {len(img_points_2D)} image points")
        return None


# Define a function to save camera calibration parameters in a text file
def save_txt(camx, dist, rvec, tvec):
    with open('info/out_v2.txt', 'w') as f:
        f.write('-- camera_matrix --\n')
        f.write(str(camx))
        f.write('\n-- dist_coeffs --\n')
        f.write(str(dist))
        f.write('\n-- rotation --\n')
        f.write(str(len(rvec)) + ' ' + str(rvec[0].shape) + ' ' + str(rvec))
        f.write('\n-- translation --\n')
        f.write(str(len(tvec)) + ' ' + str(tvec[0].shape) + ' ' + str(tvec))


# Define a function to save camera calibration parameters in a NumPy (.npz) file
def save_npz(camx, dist, rvec, tvec):
    print("duming the data into one files using numpy ")
    np.savez(
        f"info/calibration.npz",
        camMatrix=camx,
        distCoef=dist,
        rVector=rvec,
        tVector=tvec,
    )
    print("-------------------------------------------")
    print("loading data stored using numpy savez function\n \n \n")


# Define a function to check the results of the calibration
def check_calibration(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camx, dist, (w, h), 1)

    undistorted_img = cv2.undistort(img, camx, dist, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.set_title('Original Image', fontsize=50)
    ax1.imshow(img)
    ax2.set_title('Undistorted image', fontsize=50)
    ax2.imshow(undistorted_img)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Define a function to check the error rate of the calibration
def check_error_rate(mean_error=0):
    for i, (obj_point, img_point, r, t) in enumerate(zip(obj_points, img_points, rvec, tvec)):
        p_img_point, _ = cv2.projectPoints(obj_point, r, t, camx, dist)
        error = cv2.norm(img_point, p_img_point, cv2.NORM_L2) / len(p_img_point)
        mean_error += error
    print("Total error : {0}".format(mean_error / len(obj_points)))


# Define a function to load calibration parameters from a NumPy (.npz) file
def npz_load(npz_path):
    data = np.load(npz_path)
    camMatrix = data["camMatrix"]
    distCof = data["distCoef"]
    rVector = data["rVector"]
    tVector = data["tVector"]
    print('\033[1m\033[93m\bcamMatrix --- ', camMatrix, '\n\n\033[94m distCof --- ', distCof)


# Call the calibration function and get the calibration parameters
camx, dist, rvec, tvec, obj_points, img_points = calibration('images/', obj_points_3D, pattern_size, criteria)

save_npz(camx, dist, rvec, tvec)
check_calibration('images/img29.png')
check_error_rate()
