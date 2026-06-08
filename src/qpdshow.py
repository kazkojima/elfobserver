import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

#print(sys.argv)
if len(sys.argv) < 2:
    print('error: h5 file name required')
    exit()

h5file = None
imgfile = None
title_string = None
cmap = 'plasma'
for i in range(1,len(sys.argv)):
    argstr = sys.argv[i]
    if argstr.startswith('--title='):
        title_string = argstr[len('--title='):]
    elif argstr.startswith('--cmap='):
        cmap = argstr[len('--cmap='):]
    elif argstr.endswith('.h5'):
        if h5file == None:
            h5file = argstr
        else:
            print('??? extra .h5 file')
    elif argstr.endswith('.png'):
        if imgfile == None:
            imgfile = argstr
        else:
            print('??? extra .png file')
    elif argstr == '-h' or argstr == '--help':
        print('usage: python '+sys.argv[0] + ' qpd_data_file.h5 [image_file.png] --title=title_string')
        exit()
    else:
        print('extra argument: ' + argstr)

h5file = h5py.File(sys.argv[1], "r")
dset = h5file['/qpddataset']
darray = dset[:]
darray = darray[:-50,:]
tend = darray.shape[0] * 0.01
im = plt.imshow(darray.T / np.max(darray), origin='lower', extent=[0, tend, 2300, 2500], aspect='auto', cmap=cmap)
plt.xlabel('time (sec)')
plt.ylabel('curve coeff')
if title_string != None:
    plt.title(title_string)
plt.colorbar(im)
if imgfile:
    plt.savefig(imgfile)
else:
    plt.show()
