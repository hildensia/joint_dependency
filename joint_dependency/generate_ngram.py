import numpy as np
import matplotlib.pyplot as plt

n_joints = 8

angles = np.linspace(0,np.pi*2,n_joints,endpoint=False)





if n_joints % 2 == 0:
    indices = [0, 3, 1, 4, 2, 5]
    indices=[0]
    for i in range(1,n_joints):
        indices.append((indices[i-1] + n_joints/2 + ((i+1) % 2)) % n_joints)
else:
    indices=[0]
    for i in range(1,n_joints):
        indices.append((indices[i-1] + n_joints/2) % n_joints)


xs = np.cos(angles)
ys = np.sin(angles)
plt.scatter(xs,ys)

plt.plot(xs[indices],ys[indices])

print xs
print ys

plt.show()
