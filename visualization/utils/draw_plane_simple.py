import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('time')

# z = ax + by + d
x_min = 0
x_max = 10
y_min = 0
y_max = 10

a = 0
b = 10
d = 10

num_points = 50
point_size = 20

points = np.random.rand(num_points, 3)
points[:, 0] = points[:, 0]*(x_max-x_min) + x_min
points[:, 1] = points[:, 1]*(y_max-y_min) + y_min
points[:, 2] = points[:, 0]*a + points[:, 1]*b + d

mean = 0
stdev = 10
noise = np.random.normal(mean, stdev, num_points)
points[:, 2] = points[:, 2] + noise

ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size, c=(points[:, 2]),
               edgecolors='none', cmap='plasma')


# create x,y
xx, yy = np.meshgrid(range(10), range(10))
yy = yy

# calculate corresponding z
# z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
z = xx*a + yy*b + d
# plot the surface
# plt3d = plt.figure().gca(projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.2)

save_name = ("plane.png")
fig.tight_layout()
fig.savefig(save_name, dpi=600, transparent=True)
plt.close()