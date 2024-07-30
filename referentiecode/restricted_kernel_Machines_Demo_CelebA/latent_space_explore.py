import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import urllib.request
import os.path

def convert_to_imshow_format(image):
    return image.transpose(1, 2, 0)


""" Instruction for Pre-trained model: Load 'trained_RKM_LSE_h15.tar' to run experiments with h_dim=15 components.
        Loaded by default 'trained_RKM_LSE_h2.tar' to run experiments with h_dim=2 components"""


# Load model
sd_mdl = torch.load('pre_trained_model/trained_RKM_LSE_h2.tar', map_location=lambda storage, loc: storage)

net1 = sd_mdl['net1']
net3 = sd_mdl['net3']
net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
xtrain = sd_mdl['xtrain']
h = sd_mdl['h']
N = sd_mdl['N']
s = sd_mdl['s']
U = sd_mdl['U']

# Traversal along eigenvectors =========================================
m = 20

y = torch.zeros(h.shape[1], 2)
uvec = torch.zeros(h.shape[1], 2)
lambd = torch.zeros(m, 2)

for dim in range(2):
    y[:, dim] = h[int(torch.argmin(h[:, dim])), :]
    yad = torch.zeros(h.shape[1])
    yad[dim - 1] = 0.02  # adjust to stay within the cluster
    y[:, dim] = y[:, dim] + yad
    kp = h[torch.abs(h[:, dim - 1] - y[:, dim][dim - 1]) < 1e-2, :]
    ch = kp[torch.argmax(kp[:, dim])]
    l = torch.abs(ch[dim] - y[dim, dim]).detach().numpy()
    lambd[:, dim] = torch.linspace(0, float(l), steps=m)
    uvec[dim, dim] = 1  # unit vector


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'hspace': 0.5, 'wspace': 0})


def init():
    ax1.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), marker='.', color='k', s=10)
    ax3.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), marker='.', color='k', s=10)
    ax1.set_title('Along $1^{st}$ eig_vec')
    ax3.set_title('Along $2^{nd}$ eig_vec')
    plt.suptitle('Disentanglement exploration')
    return ax1, ax3


def animate(j):
    yop1 = y[:, 0] + lambd[j, 0] * uvec[:, 0]  # new vector
    yop2 = y[:, 1] + lambd[j, 1] * uvec[:, 1]  # new vector

    ax1.scatter(yop1[0].detach().numpy(), yop1[1].detach().cpu().numpy(), marker='>', color='y', s=20)
    ax3.scatter(yop2[0].detach().numpy(), yop2[1].detach().cpu().numpy(), marker='^', color='y', s=20)
    ax1.set_xticks([])
    ax3.set_xticks([])
    ax1.set_yticks([])
    ax3.set_yticks([])

    # Generate
    x_gen1 = net3(torch.mv(U, yop1)).detach().numpy()
    x_gen2 = net3(torch.mv(U, yop2)).detach().numpy()

    # Plots
    ax2.imshow(convert_to_imshow_format(x_gen1.reshape(3, 128, 128)))
    ax4.imshow(convert_to_imshow_format(x_gen2.reshape(3, 128, 128)))

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Gen. img')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('Gen. img')

    plt.pause(0.1)
    return ax1, ax2, ax3, ax4


ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, m), init_func=init)
plt.show()
