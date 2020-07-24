import numpy as np
import os

def frame_stamps_to_start_end(stamps):
    ends = list(stamps[1:])
    ends.append(ends[-1])
    se_stamps = np.stack((stamps, np.array(ends)), axis=1)
    return se_stamps

def get_frame_indices(ts, frame_ts):
    indices = [np.searchsorted(ts, fts) for fts in frame_ts]
    return np.array(indices)

def parse_crop(cropstr):
    split = cropstr.split("x")
    xsize = int(split[0])
    split = split[1].split("+")
    ysize = int(split[0])
    xoff = int(split[1])
    yoff = int(split[2])
    crop = [xoff, yoff, xoff+xsize, yoff+ysize]
    return crop

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating {directory}")
        os.makedirs(directory)

