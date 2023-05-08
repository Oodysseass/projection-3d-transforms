import numpy as np

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
phi = np.array(data['phi'])
focal = np.array(data['focal'])
