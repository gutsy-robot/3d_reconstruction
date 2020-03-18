import numpy as np
import matplotlib.pyplot as plt 
import submission 
import helper

intrinsics = np.load('../data/intrinsics.npz')
k1 = intrinsics['K1']
k2 = intrinsics['K2']

# get corresponding points
correspondences = np.load('../data/some_corresp_noisy.npz')
pts1 = correspondences['pts1']
pts2 = correspondences['pts2']

# load images
image1 = plt.imread('../data/im1.png')
image2 = plt.imread('../data/im2.png')

m = np.max(image1.shape)

# find fundamental matrix
fundamental_matrix, _ = submission.ransacF(pts1, pts2, m, 2, 0.003)
print(fundamental_matrix)
helper.displayEpipolarF(image1, image2, fundamental_matrix)
