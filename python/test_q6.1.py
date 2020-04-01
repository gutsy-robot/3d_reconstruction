import numpy as np 
import matplotlib.pyplot as plt 
import helper 
import submission
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_path = "../data/q6/" 

time_instance = 0 

data = data_path + 'time' + str(time_instance) + '.npz'

data = np.load(data)

m1, m2, m3 = data['M1'], data['M2'], data['M3']
k1, k2, k3 = data['K1'], data['K2'], data['K3']

pts1, pts2, pts3 = data['pts1'], data['pts2'], data['pts3']

c1, c2, c3 = k1.dot(m1), k2.dot(m2), k3.dot(m3)

	
# threshold = np.max(np.max(c1[:, 2]), np.max(c2[:, 2]), np.max(c3[:, 2]))

world_pts, error = submission.MultiviewReconstruction(c1, pts1, c2, pts2, c3, pts3, threshold)

# use helper's visulaise points here instead.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(world_pts[:, 0], world_pts[:, 1], world_pts[:, 2])

plt.show()