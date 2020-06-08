import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

# z = ax + by + d

x_min = 0
x_max = 100
y_min = 0
y_max = 100

a = 0
b = 10
d = 10

num_points = 5000
point_size = 10

points = np.random.rand(num_points, 3)
points[:, 0] = points[:, 0]*(x_max-x_min) + x_min
points[:, 1] = points[:, 1]*(y_max-y_min) + y_min
points[:, 2] = points[:, 0]*a + points[:, 1]*b + d

mean = 0
stdev = 10
noise = np.random.normal(mean, stdev, num_points)
points[:, 2] = points[:, 2] + noise

print(points)
new_points = points[np.where(points[:, 1] < 50)]
print(new_points)

for x in range(y_min, y_max, 1):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('time')
    ax.set_ylim([0, 100])

    new_points = points[np.where(points[:, 1] < x)]
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], s=point_size, c=(new_points[:, 2]),
               edgecolors='none', cmap='plasma')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0, c=(points[:, 2]),
               edgecolors='none', cmap='plasma')

    point = np.array([0, 1, 0])
    normal = np.array([0, 0, 1])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(100), range(10))
    yy = yy + x - 10

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    ax.plot_surface(xx, yy, z, alpha=1)

    save_name = ("frame_" + str(x) + ".png")
    fig.tight_layout()
    fig.savefig(save_name, dpi=300, transparent=True)

    # plt.show()
    plt.close()