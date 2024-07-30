import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.mixture import GaussianMixture as GMM
import urllib.request
import os.path


def convert_to_imshow_format(image):
    # convert from CHW to HWC
    return image.transpose(1, 2, 0)


# Load model
sd_mdl = torch.load('pre_trained_model/Mul_trained_RKM_F.tar',
                    map_location=lambda storage, loc: storage)

net1 = sd_mdl['net1']
net3 = sd_mdl['net3']
net2 = sd_mdl['net2']
net4 = sd_mdl['net4']
net1.load_state_dict(sd_mdl['net1_state_dict'])
net3.load_state_dict(sd_mdl['net3_state_dict'])
net2.load_state_dict(sd_mdl['net2_state_dict'])
net4.load_state_dict(sd_mdl['net4_state_dict'])
xtrain = sd_mdl['xtrain']
ytrain = sd_mdl['ytrain']
classes = sd_mdl['classes']
h = sd_mdl['h']
N = sd_mdl['N']
s = sd_mdl['s']
V = sd_mdl['V']
U = sd_mdl['U']


# Random samples from dist. over H ============================================================
gmm = GMM(n_components=1, covariance_type='full', random_state=0).fit(h.detach().cpu().numpy())
z = gmm.sample(400)
z = torch.FloatTensor(z[0])

perm2 = torch.randperm(z.shape[0])
m = 4
fig3, ax = plt.subplots(1, m)
it = 0
for i in range(1):
    for j in range(m):
        x_gen = net3(torch.mv(U, z[perm2[it], :]).cpu()).detach().numpy()
        x_gen = x_gen.reshape(3, 128, 128)
        y_gen = net4(torch.mv(V, z[perm2[it], :]).cpu()).detach().numpy()

        ax[j].imshow(convert_to_imshow_format(x_gen))
        ax[j].set_xticks([])
        ax[j].set_yticks([])

        ind = np.argpartition(y_gen, -2)[-4:]
        ax[j].set_title('{},\n {},\n {},\n {}'
                        .format(str(classes[ind[3]]), str(classes[ind[2]]), str(classes[ind[1]]),
                                str(classes[ind[0]]))
                        , fontsize=10)
        it += 1
plt.suptitle('Randomly sampled from dist. over $\mathcal{H}$')
plt.show()

# Interpolations ================================================
indx1 = 0
indx2 = 4
indx3 = 2
indx4 = 3

y1 = h[indx1, :]
y2 = h[indx2, :]
y3 = h[indx3, :]
y4 = h[indx4, :]

# # 2-D Interpolation%%%%%%%%%%%%
m = 10
T, S = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
T = np.ravel(T, order="F")
S = np.ravel(S, order="F")

fig5, ax = plt.subplots(m, m)
it = 0
for i in range(m):
    for j in range(m):

        # weights
        lambd = np.flip(np.hstack((S[it]*T[it], (1-S[it])*T[it], S[it]*(1-T[it]), (1-S[it])*(1-T[it]))), 0)
        # computation
        yop = lambd[0] * y1 + lambd[1] * y2 + lambd[2] * y3 + lambd[3] * y4

        x_gen = net3(torch.mv(U, yop).cpu()).detach().numpy()
        x_gen = x_gen.reshape(3, 128, 128)

        ax[i, j].imshow(convert_to_imshow_format(x_gen))
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        it += 1
plt.suptitle('Interpolation in latent space')
plt.show()


# 1-D Interpolation %%%%%%%%%%%%
m = 30
lambd = torch.tensor(np.linspace(0, 1, m))
fig, (ax1, ax2) = plt.subplots(1, 2)


def animate(j):
    # Computation
    yop = (1 - lambd[j]) * y1 + lambd[j] * y2
    x_gen = net3(torch.mv(U, yop).cpu()).detach().numpy()
    x_gen = x_gen.reshape(3, 128, 128)

    xtr = (1 - lambd[j]) * xtrain[indx1, :, :, :] + lambd[j] * xtrain[indx2, :, :, :]
    xtr = xtr.numpy()

    # Display
    ax1.imshow(convert_to_imshow_format(xtr))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Ground truth')

    ax2.imshow(convert_to_imshow_format(x_gen))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Latent space')
    plt.pause(0.1)
    return ax1, ax2


plt.suptitle('Interpolations')
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, m))
plt.show()
