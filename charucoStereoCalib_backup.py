import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Import the boxGrid class from the helper module
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder_path = os.path.join(current_dir, 'helper') 
sys.path.append(target_folder_path)
# import boxGrid
import bbtLocalisation.helper.boxGrid as boxGrid
# from helper import boxGrid


def calibrate_stereo_charuco(image_folder, charuco_dict=cv2.aruco.DICT_6X6_250, squares_x=5, squares_y=7, square_length=36.75, marker_length=27.6):
    """
    Calibrates a stereo camera using image pairs of Charuco boards.

    Args:
        image_folder (str): Path to folder containing stereo image pairs. 
                            Images should be named as left_*.png and right_*.png.
        charuco_dict (int): OpenCV ArUco dictionary id.
        squares_x (int): Number of squares in X direction.
        squares_y (int): Number of squares in Y direction.
        square_length (float): Square length (in meters).
        marker_length (float): Marker length (in meters).

    Returns:
        retval: RMS re-projection error.
        cameraMatrix1, distCoeffs1: Intrinsics for left camera.
        cameraMatrix2, distCoeffs2: Intrinsics for right camera.
        R, T, E, F: Stereo calibration results.
    """
    # Prepare Charuco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(charuco_dict)
    #board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)
    board = cv2.aruco.CharucoBoard((squares_y, squares_x), square_length, marker_length, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)

    # Collect image pairs
    # left_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.startswith('calib_cam0_')])
    # right_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.startswith('calib_cam1_')])

    # left_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.startswith('cam0_')]) # YW - MAC
    # right_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.startswith('cam1_')]) # YW - MAC

    left_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if 'cam0_' in f]) # YW - MAC
    right_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if 'cam1_' in f]) # YW - MAC
    assert len(left_images) == len(right_images), "Mismatched number of left/right images"

    all_corners_left, all_ids_left = [], []
    all_corners_right, all_ids_right = [], []
    #img_size = None

    for left_img_path, right_img_path in zip(left_images, right_images):
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)
        if img_left is None or img_right is None:
            continue
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        #img_size = gray_left.shape[::-1]

        """
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_left)
        if charuco_ids is not None and len(charuco_corners) > 3:
            all_corners_left.append(charuco_corners)
            all_ids_left.append(charuco_ids)
        #cv2.aruco.drawDetectedCornersCharuco(gray_left, charuco_corners, charuco_ids, (255, 0, 0))
        #cv2.imshow("Detected ArUco Markers", gray_left)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()               
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_right)
        if charuco_ids is not None and len(charuco_corners) > 3:
            all_corners_right.append(charuco_corners)
            all_ids_right.append(charuco_ids)
        """

        # Detect markers and interpolate charuco corners
        for gray, all_corners, all_ids in [(gray_left, all_corners_left, all_ids_left), (gray_right, all_corners_right, all_ids_right)]:
            #corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            if charuco_ids is not None and len(charuco_corners) > 6:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    # return length of each array within all_ids_left
    #lenEach = [len(ids) for ids in all_ids_left]
    

    #cv2.aruco.drawDetectedCornersCharuco(gray_right, charuco_corners, charuco_ids, (255, 0, 0))
    #cv2.imshow("Detected ArUco Markers", gray_right)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    

    # calibrate the two cameras separately
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.aruco.calibrateCameraCharuco(all_corners_left, all_ids_left, board, gray_left.shape, None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.aruco.calibrateCameraCharuco(all_corners_right, all_ids_right, board, gray_right.shape, None, None)    
    
    print("Left/Right Camera Calibration Results:", ret_l, ret_r)

    # Filter out pairs where detection failed
    # loop through left and right id lists for each image
    # where the left and right ids match, log the left and right corner positions for each image
    # Ugh, this is gross. Just use union1d!!!
    obj_points = []
    img_points_left = []
    img_points_right = []    
    """
    # CoPilot suggestion was gross
    for cl, il, cr, ir in zip(all_corners_left, all_ids_left, all_corners_right, all_ids_right):
        if cl is not None and cr is not None:
            corners_tmp = []
            ids_tmp = []
            points_l_tmp = []
            points_r_tmp = []
            for i_cl in il:
                if i_cl in ir:  # find the index of the left id in the right ids
                    idx = np.where(ir == i_cl)[0][0]
                    # log the left and right corners
                    ids_tmp.append(i_cl)
                    points_l_tmp.append(cl[il == i_cl])
                    points_r_tmp.append(cr[idx])

            ids_tmp = np.array(ids_tmp).reshape(-1) 
            objp, imgp_left = cv2.aruco.getBoardObjectAndImagePoints(board, points_l_tmp, ids_tmp)  # Get the object points for the corners
            objp2, imgp_right = cv2.aruco.getBoardObjectAndImagePoints(board, points_r_tmp, ids_tmp)  # Get the object points for the corners
            obj_points.append(objp)
            img_points_left.append(imgp_left)
            img_points_right.append(imgp_right)
    """ 
    for cl, il, cr, ir in zip(all_corners_left, all_ids_left, all_corners_right, all_ids_right):
        ids_lr, comm1, comm2 = np.intersect1d(il, ir, False, True)
        if len(ids_lr)<3:
            print("Not enough common markers found in left and right images.")
            continue
        cl2 = cl[comm1]
        cr2 = cr[comm2]

        objp, imgp_left = cv2.aruco.getBoardObjectAndImagePoints(board, cl2, ids_lr)  # Get the object points for the corners
        objp2, imgp_right = cv2.aruco.getBoardObjectAndImagePoints(board, cr2, ids_lr)  # Get the object points for the corners
        obj_points.append(objp)
        img_points_left.append(imgp_left)
        img_points_right.append(imgp_right)

    # Stereo calibration
    #flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        mtx_l, dist_l, mtx_r, dist_r, gray_left.shape, None, None, None, None,  # type: ignore
        None, criteria=criteria
    ) # type: ignore
    # Note that cameraMatrix1 = mtx_l and distCoeffs1 = dist_l

    # get projection matrix for each camera
    # stereoRectify is not appropriate - it assumes camera axes are parallel
    #Rcam1, Rcam2, Pcam1, Pcam2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_left.shape, R, T) #, None, None, None, None, None)

    # Manually create Projection matrices
    # see https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    # does a lot of manual steps instead of using built-in triangulation and conversion
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx_l @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx_r @ RT2 #projection matrix for C2

    # explore some other ways of calculating projection matrices
    # https://github.com/peetermrn/opencv-triangulation-example/blob/main/triangulation_example.py

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, P1, P2, R, T, E, F


def bbtFindMarkersStereo(testFiles, calibFile, family, marker_length, plotOverlay=False):
    # finds 2D image  and 3D world locations of ArUco markers in a stereo image pair

    f = open(calibFile, 'rb')
    mtx1, dist1, mtx2, dist2, P1, P2, R, T, E, F = pickle.load(f)
    f.close()

    #test_folder = '/home/nic/data/stereoTest3/'
    # load a new stereo image, detect markers and infer 3D positions
    # Load a new stereo image pair
    #left_img_path = os.path.join(test_folder, 'left_10.jpg')
    #right_img_path = os.path.join(test_folder, 'right_10.jpg')
    img_left = cv2.imread(testFiles[0])
    img_right = cv2.imread(testFiles[1]) #right_img_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(family)
    # incorporate camera calibration parameters into aruco detector
    parameters = cv2.aruco.DetectorParameters() # Create default parameters

    # Detect ArUco markers in the image
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    cnr_l, ids_l, rejected = detector.detectMarkers(gray_left)
    cnr_r, ids_r, rejected = detector.detectMarkers(gray_right)
    #rvecs, tvecs, huh = cv.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

    # get center of each marker
    cent_l = np.array([[np.mean(c[0], axis=0)] for c in cnr_l])
    cent_r = np.array([[np.mean(c[0], axis=0)] for c in cnr_r])
    # temporary - just grab the first corner of each marker
    #cent_l = np.array([[c[0][0]] for c in cnr_l])
    #cent_r = np.array([[c[0][0]] for c in cnr_r])

    # find union of ids in left and right images
    ids_lr, comm1, comm2 = np.intersect1d(ids_l, ids_r, False, True)
    cent_l2 = cent_l[comm1]
    cent_r2 = cent_r[comm2]
 
    if plotOverlay:
        # Draw detected corners on the images
        cv2.aruco.drawDetectedMarkers(gray_left, cnr_l, ids_l, (255, 0, 0))
        cv2.imshow("Left Image with aruco Corners", gray_left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Draw detected corners on the images
        cv2.aruco.drawDetectedMarkers(gray_right, cnr_r, ids_r, (255, 0, 0))
        cv2.imshow("Right Image with aruco Corners", gray_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    # Triangulate points to get 3D coordinates
    #points_left_homogeneous = cv2.convertPointsToHomogeneous(cnr_l)
    #points_right_homogeneous = cv2.convertPointsToHomogeneous(cnr_r) #points_right)

    # get projection matrix for each camera
    #Rcam1, Rcam2, Pcam1, Pcam2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, gray_left.shape, R, T) #, None, None, None, None, None)
    # We can't use stereoRectify because it assumes that camera axes are parallel

    pL = cv2.undistortPoints(cent_l2, mtx1, dist1, P=mtx1)
    pR = cv2.undistortPoints(cent_r2, mtx2, dist2, P=mtx2)
    
    #points_3d_undist = cv2.triangulatePoints(Pcam1,Pcam2, pL, pR) #points_left_3d, points_right_3d)
    #points_3d_dist = cv2.triangulatePoints(Pcam1,Pcam2, cent_l2, cent_r2) #points_left_3d, points_right_3d)
    # Convert homogeneous coordinates to 3D points
    #points_3d_undist = cv2.convertPointsFromHomogeneous(points_3d_undist.T).reshape(-1, 3)
    #points_3d_dist = cv2.convertPointsFromHomogeneous(points_3d_dist.T).reshape(-1, 3)

    points_3d_undistB = cv2.triangulatePoints(P1,P2, pL, pR) #points_left_3d, points_right_3d)
    points_3d_distB = cv2.triangulatePoints(P1,P2, cent_l2, cent_r2) #points_left_3d, points_right_3d)
    # Convert homogeneous coordinates to 3D points
    points_3d_undistB = cv2.convertPointsFromHomogeneous(points_3d_undistB.T).reshape(-1, 3)
    points_3d_distB = cv2.convertPointsFromHomogeneous(points_3d_distB.T).reshape(-1, 3)

    # math.sqrt(sum(np.square(points_3d_undistB[5]-points_3d_undistB[0])))/5
    #E, R, t = cv2.recoverPose(pL, pR, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2)

    pT_dist, _ = bbtTransform(points_3d_distB, points_3d_distB)
    pT_undist, _ = bbtTransform(points_3d_undistB, points_3d_undistB)

    # Plot the 3D points
    if plotOverlay:
        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(pT_dist[:, 0], pT_dist[:, 1], pT_dist[:, 2], 'bo')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Distorted Image')

        ax2 = fig.add_subplot(222,projection='3d')
        ax2.plot(pT_undist[:, 0], pT_undist[:, 1], pT_undist[:, 2], 'bo')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Undistorted Image')
        
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot(points_3d_distB[:, 0], points_3d_distB[:, 1], points_3d_distB[:, 2], 'bo')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Distorted Image - B')

        ax4 = fig.add_subplot(224,projection='3d')
        ax4.plot(points_3d_undistB[:, 0], points_3d_undistB[:, 1], points_3d_undistB[:, 2], 'bo')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('Undistorted Image - B')

        plt.show()

    #print("3D Points (Distorted):", points_3d_undistB)

    return points_3d_undistB, ids_lr

def bbtPlotMarkers(p3_block, p3_box):
    # Plot the 3D points
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot(p3_block[:, 0], p3_block[:, 1], p3_block[:, 2], 'bo')
    ax1.plot(p3_box[:, 0], p3_box[:, 1], p3_box[:, 2], 'ro')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('View 1')

    """ax2 = fig.add_subplot(212,projection='3d')
    ax2.plot(p3_block[:, 0], p3_block[:, 1], p3_block[:, 2], 'bo')
    ax2.plot(p3_box[:, 0], p3_box[:, 1], p3_box[:, 2], 'ro')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('View 2')
    """

    plt.show(block=True)
    return 

def bbtPlotMarkersGrid(p3_box, p3_block, bg, testFiles):

    plt = bg.plot_grid()

    # Plot the 2D points
    # should do a 3D version as a sanity check too
    plt.plot(p3_block[:, 0]+bg.block_side/2, p3_block[:, 1]+bg.block_side/2, 'm+')
    plt.plot(p3_box[:, 0]+bg.block_side/2, p3_box[:, 1]+bg.block_side/2, 'g+')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    # plt.show(block=True)
    # print("So close")
    save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"
    test_file_name = "_".join([os.path.splitext(os.path.basename(f))[0] for f in testFiles])
    unique_file_name = f'bbtMarkersGrid_{test_file_name}.png'
    plt.savefig(os.path.join(save_path, unique_file_name))

    return 


def bbtTransform(p3_box, p3_block):
    # find plane of best fit for the box points
    # rotate all marker positions so that the box points are in the XY plane

    # Fit a plane to the box points using SVD
    centroid = np.mean(p3_box, axis=0)
    points_centered = p3_box - centroid
    _, _, vh = np.linalg.svd(points_centered)
    normal = vh[-1]

    # Create rotation matrix to align normal with Z axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    if np.linalg.norm(v) < 1e-8:
        R_align = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    # Apply translation and rotation to both sets of points
    p3_box_transformed = (p3_box - centroid) @ R_align.T
    p3_block_transformed = (p3_block - centroid) @ R_align.T

    return p3_box_transformed, p3_block_transformed

def rigid_transform_3D(A, B):
    #Computes the rigid transformation (rotation and translation) that aligns points A to points B.
    #   A & B are Nx3 matrics of 3D points
    #  Returns R: 3x3 rotation matrix
    #   t: 3x1 translation vector
    
    assert A.shape == B.shape, "Input point sets must have the same shape"
    assert A.shape[1] == 3, "Input points must be 3D"
    N = A.shape[0]  # Number of points
    assert N >= 3, "At least 3 points are required to compute the transformation"
    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # Center the points
    AA = A - centroid_A 
    BB = B - centroid_B
    # Compute covariance matrix
    H = AA.T @ BB
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    # Compute rotation
    R = Vt.T @ U.T
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T      
    
    # Compute translation
    t = centroid_B - R @ centroid_A
    return R, t


# def main():
def main(calib_folder, testFolder, testFiles):

    # Calibration parameters
    # usePreCalib = False
    usePreCalib = True # True to use pre-calibrated parameters, False to run calibration
    
    # calib_folder=  'D:\\bbt\\2025\\07\\02\\calib\\' #'/mnt/d/bbt/2025/06/26/Cali/'

    # calib_folder=  '/Volumes/MNHS-MoCap/Yilin-Honours/tBBT_Image/2025/06/19/Cali' # YW - MAC
    calib_folder= calib_folder

    #calib_folder = '/home/nic/data/20250612/cali02/' # stereoCalib6/'
    calib_file = 'stereoCamParam.pckl'
    charuco_dict = cv2.aruco.DICT_6X6_250
    squares_x=5
    squares_y=7
    # A4 board
    #square_length=36.75
    #marker_length=27.6
    # A3 board
    square_length=55.2 #52.6
    marker_length=41 #42.1
    plotOverlay = False # plots marker detections for each image
    plotPoints1 = False  # plots 3D points for each image
    plotPoints2 = False
    plotPoints3 = True

    # Test parameters
    if False:
        # use this to test with one of the calibration images
        testFolder = calib_folder #'/home/nic/data/stereoCalib6/'
        testFiles = ['left_02.jpg', 'right_02.jpg']  # Example test files, should be in the same folder as the calibration images
        familyBox = cv2.aruco.DICT_6X6_250
        lenBox = 27.6
        familyBlock = cv2.aruco.DICT_6X6_250
        lenBlock = 27.6
    else: 
        # define test image pair here
        # subjID = 'TST'
        # testFolder = 'D:\\bbt\\2025\\07\\02\\'+subjID #/mnt/d/bbt/2025/06/26/1/' #'/home/nic/data/20250612/' #stereoTest6/'
        # testFiles = [subjID+'_cam0_04.png', subjID+'_cam1_04.png']  # Example test files, should be in the same folder as the calibration images

        # testFolder = '/Volumes/MNHS-MoCap/Yilin-Honours/tBBT_Image/2025/06/19/1' # YW - MAC
        # testFiles = ['cam0_04.png', 'cam1_04.png']  # YW - MAC

        testFolder = testFolder
        testFiles = testFiles  

        familyBox = cv2.aruco.DICT_6X6_50
        familyBlock = cv2.aruco.DICT_4X4_50
        lenBox = 24.8 #18.5 # (mm) aruco marker length on box
        lenBlock = 19 # 16  # (mm) aruco marker length on blocks

    
    if usePreCalib:
        # Extract and print the last two digits from the first test file name
        file_number = testFiles[0].split('_')[-1].split('.')[0]
        hand = testFiles[0].split('_')[0]
        print(hand, file_number)

        # print('Loading pre-calibrated camera parameters from ', calib_file)
    else:
        retval, mtx1, dist1, mtx2, dist2, P1, P2, R, T, E, F = calibrate_stereo_charuco(calib_folder, charuco_dict, squares_x, squares_y, square_length, marker_length)
        f = open(os.path.join(calib_folder,calib_file), 'wb')
        pickle.dump([mtx1, dist1, mtx2, dist2, P1, P2, R, T, E, F], f)
        f.close()
        print('Ran and saved camera calibration to ', calib_file)
        print('RMS re-projection error (should be <1): ', retval)
    
    # find ArUco markers in each image pair
    testFilesFull = [os.path.join(testFolder, f) for f in testFiles]
    calibFileFull = os.path.join(calib_folder, calib_file)
    p3_box, ids_lr = bbtFindMarkersStereo(testFilesFull, calibFileFull, familyBox, lenBox,plotOverlay)
    p3_block, ignor = bbtFindMarkersStereo(testFilesFull, calibFileFull, familyBlock, lenBlock,plotOverlay)

    if plotPoints1:
        bbtPlotMarkers(p3_box, p3_block)

    #p3_box_transformed, p3_block_transformed = bbtTransform(p3_box, p3_block)

    #if plotPoints2:
    #    bbtPlotMarkers(p3_box_transformed, p3_block_transformed)

    bg = boxGrid.boxGrid()
    # translate and rotate visible box markers to be in the XY plane
    #!! working here
    R,t = rigid_transform_3D(p3_box, bg.arucoBoxCenters[ids_lr])

    # Apply the transformation to the block and box markers
    p3_box2 = (R @ p3_box.T).T + t
    p3_block2 = (R @ p3_block.T).T + t

    if plotPoints3:
        bbtPlotMarkersGrid(p3_box2, p3_block2, bg, testFiles)

    # Need a way to define grid centers relative to box markers
    # for that, we'll need markers stuck to the bottom of the box in known positions
    # We can then just get 2D XY positions of the blocks (i.e. ignore Z after transformation)
    # Also need to understand how camera distortion of miscalibration is affecting us
    # Also, it would be nice to automate control of the webcams for taking calibration / test images
    #     
    # need to think about when we should be working with undisorted images/points
    # I suspect this is causing problems with estimating test positions of test markers
    # also, why is camera calibration so bad now!
    #plt.pause()
    return p3_box2, p3_block2, bg



if __name__ == "__main__":
    # calib_folder=  '/Volumes/MNHS-MoCap/Yilin-Honours/tBBT_Image/2025/06/19/Cali' # YW - MAC
    # testFolder = '/Volumes/MNHS-MoCap/Yilin-Honours/tBBT_Image/2025/06/19/1' # YW - MAC
    # testFiles = ['cam0_24.png', 'cam1_24.png']  # YW - MAC
    calib_folder=  '/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/06/19/Cali' # YW - MAC
    testFolder = '/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/06/19/CZ' # YW - MAC
    testFiles = ['left_CZ_cam0_01.png', 'left_CZ_cam0_01.png']  # YW - MAC

    # p3_box2, p3_block2, bg = main()
    p3_box2, p3_block2, bg = main(calib_folder, testFolder, testFiles)