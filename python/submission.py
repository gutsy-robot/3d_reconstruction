"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper 
import scipy.ndimage
import os

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # pass

    # normalise the points first
    pts1_normalised = pts1/M
    pts2_normalised = pts2/M

    x1 = pts1[:, 0]
    y1 = pts1[:, 1]

    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    # form matrix a.
    a = np.zeros((pts1.shape[0], 9))

    a[:, 0] = x2 * x1 
    a[:, 1] = x2 * y1
    a[:, 2] = x2
    a[:, 3] = y2 * x1
    a[:, 4] = y2 * y1  
    a[:, 5] = y2 
    a[:, 6] = x1
    a[:, 7] = y1
    a[:, 8] = np.ones(pts1.shape[0])

    # find the least square solution using SVD.
    u, s, vh = np.linalg.svd(a)
    
    # find the last singular vector
    f = vh[-1, :]
    fundamental_matrix = f.reshape(3,3)

    # impose constraint
    fundamental_matrix = helper._singularize(fundamental_matrix)

    fundamental_matrix = helper.refineF(fundamental_matrix, pts1, pts2)

    # rescale the F matrix back
    rescale_matrix = np.array([[1.0/M, 0.0, 0.0], [0, 1.0/M, 0.0], [0.0, 0.0, 1.0/M]])

    fundamental_matrix = rescale_matrix.T.dot(fundamental_matrix.dot(rescale_matrix))
    
    np.savez('q2_1.npz',F=fundamental_matrix, M=M)

    return fundamental_matrix


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    # pass

    # check the order of K1 and K2 here.
    E = K1.T.dot(F).dot(K2)
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    
    # pass
    num_pts, _ = pts1.shape

    output_pts = np.zeros((num_pts, 3))
    output_pts_homo = np.zeros((num_pts, 4))
    
    for i in range(num_pts):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]

        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        a = np.zeros((4, 4))

        a[0] = x1 * C1[2] - C1[0]
        a[1] = y1 * C1[2] - C1[1]

        a[2] = x2 * C2[2] - C2[0]
        a[3] = y2 * C2[2] - C2[1]

        u, s, vh = np.linalg.svd(a)
        p = vh[-1, :]
        p = p/p[3]
        output_pts[i] = p[0:3]
        output_pts_homo[i] = p

    projected_pts1 = C1.dot(output_pts_homo.T)
    projected_pts1 = projected_pts1/projected_pts1[-1, :]
    
    projected_pts2 = C2.dot(output_pts_homo.T)
    projected_pts2 = projected_pts2/projected_pts2[-1, :]

    projected_pts1 = projected_pts1[[0, 1], :].T
    projected_pts2 = projected_pts2[[0, 1], :].T

    err = np.sum((projected_pts1 - pts1) ** 2) + np.sum((projected_pts2-pts2) ** 2) 
    
    return output_pts, err

'''

Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # pass

    p1 = [[x1], [y1], [1]]
    p1 = np.array(p1)
   
    correspondences = np.load('../data/some_corresp.npz')
    
    epipolar_line = F.dot(p1)
    
    y = np.arange(y1 - 30, y1 + 30)
    
    x = (-(epipolar_line[1] * y + epipolar_line[2])/ epipolar_line[0])
    
    window = 5
    
    im1_gaussian = scipy.ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    im2_gaussian = scipy.ndimage.gaussian_filter(im2, sigma=1, output=np.float64)

    index = 0
    min_error = np.inf
    
    for i in range(60):
        x2 = int(x[i])
        y2 = y[i]
        
        if (x2 >= window  and x2 <= im1.shape[1] - window - 1 and y2 >= window 
            and y2 <= im1.shape[0] - window - 1):

            patch1 = im1_gaussian[y1 - window: y1 + window + 1, x1 - window: x1 + window + 1,:]
            patch2 = im2_gaussian[y2 - window: y2 + window + 1, x2 - window: x2 + window + 1,:]
            
            diff = (patch1 - patch2).flatten()
            err = (np.sum(diff**2))
            
            if (err < min_error):
                min_error = err
                index = i
    
    np.savez('q4_1.npz', F=F, pts1=correspondences['pts1'], pts2=correspondences['pts2'])
    
    return x[index],y[index]


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    pass
