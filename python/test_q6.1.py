import numpy as np 
import matplotlib.pyplot as plt 
import helper 
import submission
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image1 = plt.imread('../data/q6/cam1_time0.jpg')
image2 = plt.imread('../data/q6/cam2_time0.jpg')
image3 = plt.imread('../data/q6/cam3_time0.jpg')

data_path = "../data/q6/" 

time_instance = 0 

data = data_path + 'time' + str(time_instance) + '.npz'

data = np.load(data)

m1, m2, m3 = data['M1'], data['M2'], data['M3']
k1, k2, k3 = data['K1'], data['K2'], data['K3']

print("shape of m1 is: ", m1.shape)
print("shape of k1 is: ", k1.shape)

pts1, pts2, pts3 = data['pts1'], data['pts2'], data['pts3']

c1, c2, c3 = k1.dot(m1), k2.dot(m2), k3.dot(m3)

print("shape of pts1 initially is: ", pts1.shape)
	
# threshold = np.min(np.min(c1[:, 2]), np.min(c2[:, 2]), np.min(c3[:, 2]))
for threshold in [100]:

	world_pts, error = submission.MultiviewReconstruction(c1, pts1, c2, pts2, c3, pts3, threshold)

	print("world_pts shape: ", world_pts.shape)
	# world_pts = world_pts.reshape(12, 3)

	# use helper's visulaise points here instead.

	# helper.visualize_keypoints(image1, world_pts, 100)
	print("error is: ", np.sqrt(error))

print("shape of world_pts is: ", world_pts.shape)
helper.plot_3d_keypoint(world_pts)
# helper.visualize_keypoints(image1, pts1, 100)

# TODO ----   save the best threshold.
best_threshold = 100
fig = plt.figure()
# num_points = pts_3d.shape[0]
ax = fig.add_subplot(111, projection='3d')

connections_3d = [[0,1], [1,3], [2,3], [2,0], [4,5], [6,7], [8,9], [9,11], [10,11], [10,8], [0,4], [4,8], [1,5], 
[5,9], [2,6], [6,10], [3,7], [7,11]]
colors = ['blue','blue','blue','blue','red','magenta','green','green','green','green','red','red','red','red','magenta','magenta','magenta','magenta']

for k  in range(0, 10):
	# fig, ax = plt.subplots()
	im1 = plt.imread('../data/q6/cam1_time' + str(k) + '.jpg')
	im2 = plt.imread('../data/q6/cam2_time' + str(k) + '.jpg')
	im3 = plt.imread('../data/q6/cam3_time' + str(k) + '.jpg')

	data = data_path + 'time' + str(k) + '.npz'
	data = np.load(data)
	m1, m2, m3 = data['M1'], data['M2'], data['M3']
	k1, k2, k3 = data['K1'], data['K2'], data['K3']
	pts1, pts2, pts3 = data['pts1'], data['pts2'], data['pts3']
	c1, c2, c3 = k1.dot(m1), k2.dot(m2), k3.dot(m3)

	world_pts, error = submission.MultiviewReconstruction(c1, pts1, c2, pts2, c3, pts3, best_threshold)
	pts_3d = world_pts
	for j in range(len(connections_3d)):
		index0, index1 = connections_3d[j]	
		xline = [pts_3d[index0,0], pts_3d[index1,0]]
		yline = [pts_3d[index0,1], pts_3d[index1,1]]
		zline = [pts_3d[index0,2], pts_3d[index1,2]]
		ax.plot(xline, yline, zline, color=colors[j])
	np.set_printoptions(threshold=1e6, suppress=True)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
	# print("world_pts shape: ", world_pts.shape)
