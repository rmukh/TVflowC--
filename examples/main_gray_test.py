import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tv_flow_python

print("prepare variables for TV flow")
T = 3 # diffusion time
dt = 1 / 50 # diffusion step

NOB = int(np.ceil(T / dt))

NIT = 100 # number of iteration for TV denoising convergence
tol = 1e-5 # convergence tolerance

lami = 2 * dt

im_raw = Image.open('fruits.bmp')
f = im_raw.convert('L') # to gray-scale
f = np.array(f)

n, m = f.shape

# Normalize image
f = f / 255.0
f = 5.0 * f

f_F = np.asfortranarray(f) # to Eigen3 format 

print("Run...")
# Measure an execution time of the code below
start = time.time()
S = tv_flow_python.run_TV_flow(f_F, n, m, NOB, lami, dt, tol, NIT)
end = time.time()

print("Specter values", S)
print('Time: ', end - start)

t = 2 * dt + np.arange(NOB) * dt

# Plot spectral response
plt.plot(t, S)
plt.ylabel('$\sum|\phi(n)|$')
plt.xlabel('$t$')

plt.show()