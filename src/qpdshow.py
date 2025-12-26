import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

#print(sys.argv)
if len(sys.argv) < 2:
    print('error: h5 file name required')
    exit()

h5file = h5py.File(sys.argv[1], "r")
dset = h5file['/qpddataset']
darray = dset[:]
darray = darray[:-50,:]
tend = darray.shape[0] * 0.01
im = plt.imshow(darray.T / np.max(darray), origin='lower', extent=[0, tend, 2300, 2500], aspect='auto', cmap='plasma')
plt.xlabel('time (sec)')
plt.ylabel('curve coeff')
plt.title('normalized QPD magnitude\n')
plt.colorbar(im)
plt.show()
