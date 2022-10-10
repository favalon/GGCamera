import numpy as np

norm_M1 = np.load("GAN_train/norm/normdata_r6.npy")
rand5=np.load("GAN_train/norm/rand_input5.npy")
randzeros=np.zeros((1, 1, 6, 35))


rotnorm=np.load("GAN_train/norm/RoTtrainorm.npy")
