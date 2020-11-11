import json
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from math import fabs, ceil, floor
from torch.nn import ZeroPad2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv


def ensure_dir(dirname):
    """
    Ensure a directory exists, if not create it
    @param dirname Directory name
    @returns None
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0:self.iy1, self.ix0:self.ix1]


def format_power(size):
    power = 1e3
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]

def plot_image(image, lognorm=False, cmap='gray', bbox=None, ticks=False, norm=True, savename=None, colorbar=False):
    """
    Plot an image
    :param image: The image to plot, as np array
    :param lognorm: If true, apply log transform the normalize image
    :param cmap: Colormap (defaul gray)
    :param bbox: Optional bounding box to draw on image, as array with [[top corner x,y,w,h]]
    :param ticks: Whether or not to draw axis ticks
    :param norm: Normalize image?
    :param savename: Optional save path
    :param colorbar: Display color bar if true
    """
    fig, ax = plt.subplots(1)
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    if norm:
        image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX)
    ims = ax.imshow(image, cmap=cmap)
    if bbox is not None:
        w,h = bbox[2], bbox[3]
        rect = patches.Rectangle((bbox[0:2]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if colorbar:
        fig.colorbar(ims)
    if not ticks:
        plt.axis('off')
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plot_image_grid(images, grid_shape=None, lognorm=False,
        cmap='gray', bbox=None, norm=True, savename=None,
        colorbar=False):
    """
    Given a list of images, stitches them into a grid and displays/saves the grid
    @param images List of images
    @param grid_shape Shape of the grid
    @param lognorm Logarithmic normalise the image
    @param cmap Color map to use
    @param bbox Draw a bounding box on the image
    @param norm If True, normalise the image
    @param savename If set, save the image to that path
    @param colorbar If true, plot the colorbar
    """
    if grid_shape is None:
        grid_shape = [1, len(images)]

    col = []
    img_idx = 0
    for xc in range(grid_shape[0]):
        row = []
        for yc in range(grid_shape[1]):
            image = images[img_idx]
            if lognorm:
                image = np.log10(image)
                cmap='viridis'
            if norm:
                image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX)
            row.append(image)
            img_idx += 1
        col.append(np.concatenate(row, axis=1))
    comp_img = np.concatenate(col, axis=0)
    if savename is None:
        plot_image(comp_img, norm=False, colorbar=colorbar, cmap=cmap)
    else:
        save_image(comp_img, fname=savename, colorbar=colorbar, cmap=cmap)

def save_image(image, fname=None, lognorm=False, cmap='gray', bbox=None, colorbar=False):
    fname = "/tmp/img.png" if fname is None else fname
    fig, ax = plt.subplots(1)
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX)
    ims = ax.imshow(image, cmap=cmap)
    if bbox is not None:
        w = bbox[1][0]-bbox[0][0]
        h = bbox[1][1]-bbox[0][1]
        rect = patches.Rectangle((bbox[0]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if colorbar:
        fig.colorbar(ims)
    plt.savefig(fname, dpi=150)
    plt.close()

def flow2bgr_np(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339
    @param disp_x A [H x W] NumPy array containing the X displacement
    @param disp_y A [H x W] NumPy array containing the Y displacement
    @returns A [H x W x 3] NumPy array containing a color-coded representation of the flow [0, 255]
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    # X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    # flow_x = (X - disp_x) * float(W) / 2
    # flow_y = (Y - disp_y) * float(H) / 2
    # magnitude, angle = cv.cartToPolar(flow_x, flow_y)
    # magnitude, angle = cv.cartToPolar(disp_x, disp_y)

    # follow alex zhu color convention https://github.com/daniilidis-group/EV-FlowNet

    flows = np.stack((disp_x, disp_y), axis=2)
    magnitude = np.linalg.norm(flows, axis=2)

    angle = np.arctan2(disp_y, disp_x)
    angle += np.pi
    angle *= 180. / np.pi / 2.
    angle = angle.astype(np.uint8)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    else:
        v = np.clip(255.0 * magnitude / max_magnitude, 0, 255)
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle
    hsv[..., 2] = v
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr
