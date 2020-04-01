import numpy as np
import matplotlib.pyplot as plt 
import submission 
import helper

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

# get camera matrices
intrinsics = np.load('../data/intrinsics.npz')
k1 = intrinsics['K1']
k2 = intrinsics['K2']

# get corresponding points
correspondences = np.load('../data/some_corresp.npz')
pts1 = correspondences['pts1']
pts2 = correspondences['pts2']

# load images
image1 = plt.imread('../data/im1.png')
image2 = plt.imread('../data/im2.png')

# get scale factor
m = np.max(image1.shape)

# find fundamental matrix
fundamental_matrix = submission.eightpoint(pts1, pts2, m)

# get essential matrix
essential_matrix = submission.essentialMatrix(fundamental_matrix, k1, k2)

# get projective matrix for camera1. 
extrinsic_mat1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
c1 = k1.dot(extrinsic_mat1)

# get possible m2 solutions
m2s = helper.camera2(essential_matrix)

num_m2s = m2s.shape[2]

for i in range(num_m2s):
	
	# get projective matrix for camera2.
	m2 = m2s[:, :, i]
	c2 = k2.dot(m2)

	# get the world coordinates of the 3D points.
	p, error = submission.triangulate(c1, pts1, c2, pts2)

	# check validity of solution.
	if (np.all(p[:,2] > 0)) :
            break


np.savez('q3_3.npz', M2=m2, C2=c2, P=p)

# return m1, c1, m2, c2, fundamental_matrix

helper.epipolarMatchGUI(image1, image2, fundamental_matrix)
print("essential matrix is: ", essential_matrix)
# helper.displayEpipolarF(image1, image2, fundamental_matrix)

