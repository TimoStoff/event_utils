import h5py
import numpy as np

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def binary_search_h5_timestamp(hdf_path, l, r, x, side='left'):
    f = h5py.File(hdf_path, 'r')
    return binary_search_h5_dset(f['events/ts'], x, l=l, r=r, side=side)
