"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper 
import scipy.ndimage
import os
import scipy.optimize 

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
    pts1_normalised = pts1 /float(M)
    pts2_normalised = pts2 /float(M)

    x1 = pts1_normalised[:, 0]
    y1 = pts1_normalised[:, 1]

    x2 = pts2_normalised[:, 0]
    y2 = pts2_normalised[:, 1]

    # form matrix a.
    a = np.zeros((pts1.shape[0], 9))

    a[:, 0] = x2 * x1 
    # print("shape: ", a[:, 0].shape)
    a[:, 1] = x2 * y1
    a[:, 2] = x2
    a[:, 3] = y2 * x1
    a[:, 4] = y2 * y1  
    a[:, 5] = y2 
    a[:, 6] = x1
    a[:, 7] = y1
    a[:, 8] = np.ones(pts1.shape[0])

    # print("shape of a is: ", a.shape)
    # find the least square solution using SVD.
    u, s, vh = np.linalg.svd(a)
    
    # find the last singular vector
    f = vh[-1, :]
    fundamental_matrix = f.reshape(3,3)

    # impose constraint
    fundamental_matrix = helper._singularize(fundamental_matrix)

    fundamental_matrix = helper.refineF(fundamental_matrix, 
        pts1_normalised, pts2_normalised)

    # rescale the F matrix back
    rescale_matrix = np.array([[1.0/M, 0.0, 0.0], [0, 1.0/M, 0.0], [0.0, 0.0, 1.0]])

    fundamental_matrix = rescale_matrix.T.dot((fundamental_matrix.dot(rescale_matrix)))
    
    np.savez('q2_1.npz', F=fundamental_matrix, M=M)

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
    E = K2.T.dot(F).dot(K1)
    
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
    
    # tune these 
    y = np.arange(y1 - 25, y1 + 25)
    
    x = (-(epipolar_line[1] * y + epipolar_line[2])/ epipolar_line[0])
    
    # tune these
    window = 6
    
    im1_gaussian = scipy.ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    im2_gaussian = scipy.ndimage.gaussian_filter(im2, sigma=1, output=np.float64)

    index = 0
    min_error = np.inf
    
    for i in range(50):
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
    # pass

    pts1_normalised = pts1/M
    pts2_normalised = pts2/M

    pts1_h = np.vstack((pts1.T, np.ones(pts1.shape[0]))).T
    pts2_h = np.vstack((pts2.T, np.ones(pts2.shape[0]))).T

    # print("pts1 shape homo: ", pts1_h.shape)
    # print("pts1 homo 0: ", pts1_h[0])
    # print("pts1 normal 0: ", pts1[0])
    
    error = np.inf
    min_error = np.inf
    iters = 0

    best_inlier_count = 0
    inliers1 = None
    inliers2 = None
    inlier_ind = None
    while iters < nIters:
        indices = np.random.randint(0, pts1.shape[0], 8)
        # print(pts1[indices, :].shape)
        fundamental_matrix = eightpoint(pts1[indices, :], pts2[indices, :], M)

        # compute the inliers and outliers
        # mapped_pts = fundamental_matrix.dot(pts1_h)

        # pts1_dh = mapped_pts / mapped_pts[-1, :]
        # pts1_dh = pts1_dh.Tr

        # pts1_dh = pts1[:, [0, 1]]

        error = pts2_h.T * (fundamental_matrix.dot(pts1_h.T))
        # print(error[:, 0])
        error = abs(np.sum(error, axis=0))
        # print("shape of error is: ", error.shape)
        # print(error[0])
        # print(pts2_h[0].dot(fundamental_matrix.dot(pts1_h[0].reshape(3, 1))))
        
        # print("======")

        # print(pts2_h[0] * fundamental_matrix.dot(pts1_h.T)[:, 0])        
        # print(pts2_h[0].dot(fundamental_matrix.dot(pts1_h[0].reshape(3, 1))))

        num_inlers = np.sum(error < tol)

        if num_inlers >= best_inlier_count:
            best_inlier_count = num_inlers
            ind = np.argwhere(error < tol).flatten()
            
            inliers1 = pts1[ind, :]
            inliers2 = pts2[ind, :]
            inlier_ind = ind

        iters +=1 
        # print("completed one iteration...")

    # print("shape of inliers1 is: ", inliers1.shape)
    # print("shape of inliers2 is: ", inliers2.shape)

    fundamental_matrix = eightpoint(inliers1, inliers2, M)

    # print("shape of fundamental_matrix is: ", fundamental_matrix.shape)

    inlier_bool = np.ones(pts1.shape[0]) > 2
    # inlier_bool = 
    inlier_bool[inlier_ind] = True
    # print(inlier_bool)
    # print(inlier_ind)
    return fundamental_matrix, inlier_bool

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    # pass
    t = np.linalg.norm(r)
    
    if t == 0:
        return np.identity(3)

    else:
        a = r / t
        a1 = a[0,0]
        a2 = a[1,0]
        a3 = a[2,0]
        cross = np.array([[0, -a3, a2],[a3, 0, -a1],
                          [-a2, a1, 0]])
        R = np.identity(3) * np.cos(t) + (1 - np.cos(t)) * a.dot(a.T) + np.sin(t) * cross
        return R 
    
'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    # pass
    rho_matix = (R - R.T)/2
    rho = np.array([[-rho_matix[1,2]],[rho_matix[0,2]],[rho_matix[1,0]]])
    sin = np.linalg.norm(rho)

    # trace is cos thetha.
    cos = (R[0,0] + R[1,1] + R[2,2] - 1)/ 2
    
    if (sin != 0):
        u = rho / sin
        t = np.arctan2(sin,cos)
        rotation_vector =  t * u
        return rotation_vector

    if (sin == 0 and cos == 1):
        return np.zeros((3,1))
    
    if (sin == 0 and cos == -1):
        uu_t = R + np.eye(3)
        if (np.linalg.norm(uu_t[:,0]) != 0):
            v = uu_t[:,0]
        
        elif(np.linalg.norm(uu_t[:,1]) != 0):
            v = uu_t[:,1]
        
        elif(np.linalg.norm(uu_t[:,2]) != 0):
            v = uu_t[:,2]
        
        u = v / np.linalg.norm(v)
        
        rotation_vector = u * np.pi
        
        r1 = rotation_vector[0,0]
        r2 = rotation_vector[1,0]
        r3 = rotation_vector[2,0]
        
        if ((r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2<0) or (r1<0)):
            rotation_vector =  -rotation_vector
            
        return r
    

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
    # pass

    c1 = K1.dot(M1)

    num_points = p1.shape[0]

    p = x[0:3 * num_points].reshape((num_points, 3))
    r2 = x[3 * num_points: 3 * num_points + 3].reshape((3, 1))
    t2 = x[3 * num_points + 3 : ].reshape((3, 1))

    rotation_matrix2 = rodrigues(r2)
    c2 = K2.dot(np.hstack((rotation_matrix2, t2)))

    p_homogenous = np.hstack((p,np.ones((num_points,1))))

    projected1_homogenous = c1.dot(p_homogenous.T)

    projected1 = projected1_homogenous / projected1_homogenous[-1, :]

    projected2_homogenous = c2.dot(p_homogenous.T)

    projected2 = projected2_homogenous / projected2_homogenous[-1, :]

    p1_hat = projected1.T[:, 0:2]
    p2_hat = projected2.T[:, 0:2]

    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), 
        (p2 - p2_hat).reshape([-1])]).reshape(4 * num_points,1)

    return residuals


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
    # pass

    R2_initial = M2_init[:,:3]
    rot2_initial = invRodrigues(R2_initial).flatten()
    trans2_initial = M2_init[:,3]
    
    f = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()

    num_points = p1.shape[0]
    
    x0 = np.hstack((P_init.flatten(), rot2_initial, trans2_initial))

    print("error initial:",np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x0)))
    
    res = scipy.optimize.minimize(f, x0)
    
    x = res.x

    print("error optmzed:",np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x)))


    P2 = x[0 : 3 * num_points].reshape((num_points, 3))
    r2 = x[3 * num_points : 3 * num_points + 3].reshape((3, 1))
    t2 = x[3 * num_points + 3:].reshape((3, 1))
    
    rot_final = rodrigues(r2)
    extrinsics2_final = np.hstack((rot_final, t2))

    return extrinsics2_final, P2


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
    # pass

    # traingulate using all 3 views and get the 3d point. 

    # calculate residuals for all three pairs of points. 
    # find the pair with the lowest residual error. 

    # run bundle adjustment on that. 
    # Return the optimised points and the error.

    # other approach do bundleAdjustment for all.

    # the new bundleAdjustment function should return only the optimised points.
    # the rodriguez residual function should take (C1, pts1, C2, pts2, C3, pts3, 3d points) and return a residual.
    ind = np.argmin(np.array([np.sum(pts1[:, 2] > Thres), np.sum(pts2[:, 2] > Thres), np.sum(pts3[:, 2] > Thres)]))

    if ind == 0:
        ind = pts1[:, 2] > Thres
    elif ind == 1:
        ind = pts2[:, 2] > Thres

    elif ind == 2:
        ind = pts3[:, 2] > Thres

    pts1, pts2, pts3 = pts1[ind][:, [0, 1]], pts2[ind][: , [0, 1]], pts3[ind][:, [0, 1]]

    num_pts, _ = pts1.shape
    # print("num_pts is: ", num_pts)

    output_pts = np.zeros((num_pts, 3))
    output_pts_homo = np.zeros((num_pts, 4))
    
    for i in range(num_pts):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]

        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        x3 = pts3[i, 0]
        y3 = pts3[i, 1]

        a = np.zeros((6, 4))

        a[0] = x1 * C1[2] - C1[0]
        a[1] = y1 * C1[2] - C1[1]

        a[2] = x2 * C2[2] - C2[0]
        a[3] = y2 * C2[2] - C2[1]

        a[4] = x3 * C3[2] - C3[0]
        a[5] = y3 * C3[2] - C3[0]

        u, s, vh = np.linalg.svd(a)
        p = vh[-1, :]
        p = p/p[3]
        output_pts[i] = p[0:3]
        output_pts_homo[i] = p

    # print("shape of output_pts is: ", output_pts.shape)
    # print("initial triangulation successful")
    
    pts = bundleAdjustment_multiview(C1, pts1, C2, pts2, C3, pts3, output_pts)
    # print("passed bundle")
    error = rodriguesResidual_multiview(C1, pts1, C2, pts2, C3, pts3, pts)

    error = (error ** 2).sum()
    pts = pts.reshape(pts1.shape[0], 3)

    return pts, error

def bundleAdjustment_multiview(C1, p1, C2,  p2, C3, p3, P_init):
    # Replace pass by your implementation
    
    # print("shape of p1 is: ", p1.shape)
    f = lambda x: (rodriguesResidual_multiview(C1, p1, C2, p2, C3, p3, x)**2).sum()

    num_points = p1.shape[0]
    
    x0 = P_init    
    # print("shape of x0 is: ", x0.shape)
    # print("x0 is: ", x0)
    print("error before bundleadjestment: ", np.sqrt((rodriguesResidual_multiview(C1, p1, C2, p2, C3, p3, x0)**2).sum()))
    res = scipy.optimize.minimize(f, x0)
    
    x = res.x

    return x


def rodriguesResidual_multiview(C1, p1, C2, p2, C3, p3, p):
    # Replace pass by your implementation
    # pass

    # c1 = K1.dot(M1)
    
    # print("p is: ", p)
    num_points = p1.shape[0]
    p = p.reshape(num_points, 3)
    # print("reached here..")
    # print("shape of p is: ", p.shape)
    # print("shape of ones is: ", np.ones((num_points,1)).shape)

    p_homogenous = np.hstack((p,np.ones((num_points,1))))

    # print("passed hstack")

    projected1_homogenous = C1.dot(p_homogenous.T)
    projected1 = projected1_homogenous / projected1_homogenous[-1, :]

    projected2_homogenous = C2.dot(p_homogenous.T)
    projected2 = projected2_homogenous / projected2_homogenous[-1, :]

    projected3_homogenous = C3.dot(p_homogenous.T)
    projected3 = projected3_homogenous / projected3_homogenous[-1, :]


    p1_hat = projected1.T[:, 0:2]
    p2_hat = projected2.T[:, 0:2]

    p3_hat = projected3.T[:, 0:2]

    # print("shape of p2hat is: ", p2_hat.shape)
    # print("shape of p2 is: ", p2.shape)
    
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), 
        (p2 - p2_hat).reshape([-1]), (p3 - p3_hat).reshape([-1])]).reshape(6 * num_points,1)

    return residuals
