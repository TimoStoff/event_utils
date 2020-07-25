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

def crop_to_size(crop):
    return [crop[0]-crop[1], crop[2]-crop[3]]

def parse_crop(cropstr):
    """
    Crop is provided as string, same as imagemagick:
        size_x, size_y, offset_x, offset_y, eg 10x10+30+30 would cut a 10x10 square at 30,30
    Output is the indices as would be used in a numpy array. In the example,
    [30,40,30,40] (ie [miny, maxy, minx, maxx])

    """
    split = cropstr.split("x")
    xsize = int(split[0])
    split = split[1].split("+")
    ysize = int(split[0])
    xoff = int(split[1])
    yoff = int(split[2])
    crop = [yoff, yoff+ysize, xoff, xoff+xsize]
    return crop

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating {directory}")
        os.makedirs(directory)

