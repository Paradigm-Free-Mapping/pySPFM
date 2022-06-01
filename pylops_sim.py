import matplotlib.pyplot as plt
import numpy as np
import pylops
import pyproximal
from pyproximal.optimization.primal import AcceleratedProximalGradient
from pyproximal.proximal import L2, L21_plus_L1
from splora.deconvolution.fista import fista
from splora.deconvolution.hrf_matrix import hrf_linear

from pySPFM.deconvolution.select_lambda import select_lambda

nt = 300
nv = 1000
tr = 2

y_ideal = np.zeros((nt, nv))

y_ideal[50:51, :] = 1
y_ideal[200:205, :] = 1
y_ideal[250:252, :] = 1

p = [6, 16, 1, 1, 6, 0, 32]
hrf_SPM = hrf_linear(2, p)

filler = np.zeros(nt - hrf_SPM.shape[0], dtype=int)
hrf_SPM = np.append(hrf_SPM, filler)

temp = hrf_SPM

for i in range(nt - 1):
    foo = np.append(np.zeros(i + 1), hrf_SPM[0 : (len(hrf_SPM) - i - 1)])
    temp = np.column_stack((temp, foo))

hrf_matrix = temp

hrf = pylops.MatrixMult(hrf_matrix)
# hrf = pylops.signalprocessing.Convolve1D(N=nt, h=hrf_SPM)

y = hrf.dot(y_ideal)
y_noise = y + np.repeat(np.random.normal(0, 0.2, y.shape[0])[:, np.newaxis], nv, axis=1)
y = y_noise.copy()

breakpoint()
np.save("sim_data.npy", y)
np.save("sim_hrf.npy", hrf_matrix)

lambda_ = select_lambda(hrf_matrix, y, criteria="ut")[0]
print(lambda_)

tau = 1 / (np.linalg.norm(hrf_matrix) ** 2)

l21_l1 = L21_plus_L1(sigma=lambda_, rho=0.8)
l2 = L2(Op=hrf, b=y_noise)
fista_results = AcceleratedProximalGradient(
    l2,
    l21_l1,
    tau=tau,
    x0=np.zeros((nt, nv)),
    epsg=np.ones(nv),
    niter=400,
    acceleration="fista",
    show=False,
)

np.save("pylops_fista.npy", fista_results)
