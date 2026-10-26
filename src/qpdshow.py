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
marking = False
accum = False
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
    elif argstr == '--acc':
        accum = True
    elif argstr == '--mark':
        marking = True
    elif argstr == '-h' or argstr == '--help':
        print('usage: python '+sys.argv[0] + ' qpd_data_file.h5 [image_file.png] [--title=title_string] [--cmap=cmap_name] [--mark]')
        exit()
    else:
        print('extra argument: ' + argstr)

if h5file == None:
    print('no h5 file')
    exit()

h5file = h5py.File(h5file, "r")
dset = h5file['/qpddataset']
darray = dset[:]
darray = darray[:-50,:]
tend = darray.shape[0] * 0.01
na = darray.T / np.max(darray)
if accum:
    la = np.sum(na, axis=0)
    y = la / np.max(la)
    x = np.arange(0, tend, 0.01)
    plt.plot(x, y)
    plt.ylabel('mag')
else:
    im = plt.imshow(na, origin='lower', extent=[0, tend, 2300, 2500], aspect='auto', cmap=cmap)
    if marking:
        max_row, max_col = np.unravel_index(np.argmax(na), na.shape)
        max_x = (max_col / na.shape[1]) * tend
        max_y = max_row + 2300
        plt.scatter(max_x, max_y, color="white", marker="x", s=30, linewidths=1, label="")
        plt.colorbar(im)
        plt.ylabel('curve coeff')

plt.xlabel('time (sec)')
if title_string != None:
    plt.title(title_string)
if imgfile:
    plt.savefig(imgfile)
else:
    plt.show()
