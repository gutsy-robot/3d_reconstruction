import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission, helper
import helper

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
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
# M1, C1, M2, C2, F = findM2.findM2()

data = np.load('../data/templeCoords.npz')
# im1 = plt.imread('../data/im1.png')
# im2 = plt.imread('../data/im2.png')
x1 = data['x1']
y1 = data['y1']

print("shape of x1 is: ", x1.shape)

pts1 = np.zeros((x1.shape[0], 2))
pts1[:, 0] = x1.flatten()
pts1[:, 1] = y1.flatten()

pts2 = np.zeros((x1.shape[0], 2))

for i in range(x1.shape[0]):

    x2, y2 = submission.epipolarCorrespondence(image1, image2, fundamental_matrix, 
    	x1[i, 0], y1[i, 0])
    
    pts2[i, 0] = x2
    pts2[i, 1] = y2
    

object_coords, error = submission.triangulate(c1, pts1, c2, pts2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(object_coords[:, 0], object_coords[:, 1], object_coords[:, 2])

plt.show()
np.savez('q4_2.npz', F=fundamental_matrix, M1=m1, M2=m2, C1=c1, C2=c2)