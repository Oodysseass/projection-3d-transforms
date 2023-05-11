import numpy as np
from matplotlib import pyplot as plt
from functions import RenderObject, RotateTranslate

# get data
data = np.load("h2.npy", allow_pickle=True).item()
verts3d = np.array(data['verts3d'])
vcolors = np.array(data['vcolors'])
faces = np.array(data['faces'])
c_org = np.array(data['c_org'])
c_lookat = np.array(data['c_lookat'])
c_up = np.array(data['c_up'])
t_1 = np.array(data['t_1'])
t_2 = np.array(data['t_2'])
u = np.array(data['u'])
phi = data['phi']
focal = data['focal']

H = W = 15
Rows = Columns = 512
f = 70

# starting image
img = RenderObject(verts3d, faces, vcolors, H, W, \
                   Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('0.jpg', img)
#plt.imshow(img)
#plt.show()

# move by t1 and don't rotate
verts3d = RotateTranslate(verts3d, 0, np.ones(3), np.zeros(3), t_1)

img = RenderObject(verts3d, faces, vcolors, H, W, \
                   Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('1.jpg', img)
#plt.imshow(img)
#plt.show()

# rotate by phi around u and dont rotate
verts3d = RotateTranslate(verts3d, phi, u, np.zeros(3), np.zeros(3))

img = RenderObject(verts3d, faces, vcolors, H, W, \
                   Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('2.jpg', img)
#plt.imshow(img)
#plt.show()

# move by t2 and don't rotate
verts3d = RotateTranslate(verts3d, 0, np.ones(3), np.zeros(3), t_2)

img = RenderObject(verts3d, faces, vcolors, H, W, \
                   Rows, Columns, f, c_org, c_lookat, c_up)
plt.imsave('3.jpg', img)
#plt.imshow(img)
#plt.show()