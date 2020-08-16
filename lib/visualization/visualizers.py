import numpy as np
import numpy.lib.recfunctions as nlr
import cv2 as cv
from skimage.measure import block_reduce
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..representations.image import events_to_image, TimestampImage
from ..representations.voxel_grid import events_to_voxel
from ..util.event_util import clip_events_to_bounds
from .visualization_utils import *
from tqdm import tqdm

class Visualizer():

    def __init__(self):
        raise NotImplementedError

    def plot_events(self, xs, ys, ts, ps, save_path, **kwargs):
        raise NotImplementedError

class TimeStampImageVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.ts_img = TimestampImage(sensor_size)
        self.sensor_size = sensor_size

    def plot_events(self, xs, ys, ts, ps, save_path, **kwargs):
        self.ts_img.add_events(xs, ys, ts, ps)
        timestamp_image = self.ts_img.get_image()
        fig = plt.figure()
        plt.imshow(timestamp_image, cmap='viridis')
        plt.show()

class EventImageVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

    def plot_events(self, xs, ys, ts, ps, save_path, **kwargs):
        img = events_to_image(xs.astype(int), ys.astype(int), ps, self.sensor_size, interpolation=None, padding=False)
        mn, mx = np.min(img), np.max(img)
        img = (img-mn)/(mx-mn)

        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()


#def plot_events(xs, ys, ts, ps, save_path=None, num_compress='auto', num_show=1000,
#        event_size=2, elev=0, azim=45, imgs=[], img_ts=[], show_events=True,
#        show_frames=True, show_plot=False, crop=None, compress_front=False,
#        marker='.', stride = 1, invert=False, img_size=None, show_axes=False):
#    """
#    Given events, plot these in a spatiotemporal volume.
#    :param: xs x coords of events
#    :param: ys y coords of events
#    :param: ts t coords of events
#    :param: ps p coords of events
#    :param: save_path if set, will save plot to here
#    :param: num_compress will take this number of events from the end
#        and create an event image from these. This event image will
#        be displayed at the end of the spatiotemporal volume
#    :param: num_show sets the number of events to plot. If set to -1
#        will plot all of the events (can be potentially expensive)
#    :param: event_size sets the size of the plotted events
#    :param: elev sets the elevation of the plot
#    :param: azim sets the azimuth of the plot
#    :param: imgs a list of images to draw into the spatiotemporal volume
#    :param: img_ts a list of the position on the temporal axis where each
#        image from 'imgs' is to be placed (the timestamp of the images, usually)
#    :param: show_events if False, will not plot the events (only images)
#    :param: crop a list of length 4 that sets the crop of the plot (must
#        be in the format [top_left_y, top_left_x, height, width]
#    """
#    #Crop events
#    if img_size is None:
#        img_size = [max(ys), max(ps)] if len(imgs)==0 else imgs[0].shape[0:2]
#    crop = [0, img_size[0], 0, img_size[1]] if crop is None else crop
#    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
#    xs, ys = xs-crop[2], ys-crop[0]
#
#    #Defaults and range checks
#    num_show = len(xs) if num_show == -1 else num_show
#    skip = max(len(xs)//num_show, 1)
#    num_compress = len(xs) if num_compress == -1 else num_compress
#    num_compress = min(img_size[0]*img_size[1]*0.5, len(xs)) if num_compress=='auto' else num_compress
#    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]
#
#    #Prepare the plot, set colors
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
#    colors = ['r' if p>0 else ('#00DAFF' if invert else 'b') for p in ps]
#
#    #Plot images
#    if len(imgs)>0 and show_frames:
#        for imgidx, (img, img_ts) in enumerate(zip(imgs, img_ts)):
#            img = img[crop[0]:crop[1], crop[2]:crop[3]]
#            if len(img.shape)==2:
#                img = np.stack((img, img, img), axis=2)
#            if num_compress > 0:
#                events_img = events_to_image(xs[0:num_compress], ys[0:num_compress],
#                        np.ones(num_compress), sensor_size=img.shape[0:2])
#                events_img[events_img>0] = 1
#                img[:,:,1]+=events_img[:,:]
#                img = np.clip(img, 0, 1)
#            x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
#            event_idx = np.searchsorted(ts, img_ts)
#
#            ax.scatter(xs[0:event_idx], ts[0:event_idx], ys[0:event_idx], zdir='z',
#                    c=colors[0:event_idx], facecolors=colors[0:event_idx],
#                    s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
#
#            ax.plot_surface(y, img_ts, x, rstride=stride, cstride=stride, facecolors=img, alpha=1)
#
#            ax.scatter(xs[event_idx:-1], ts[event_idx:-1], ys[event_idx:-1], zdir='z',
#                    c=colors[event_idx:-1], facecolors=colors[event_idx:-1],
#                    s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
#
#    elif num_compress > 0:
#        # Plot events
#        ax.scatter(xs[::skip], ts[::skip], ys[::skip], zdir='z', c=colors[::skip], facecolors=colors[::skip],
#                s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
#        num_compress = min(num_compress, len(xs))
#        if not compress_front:
#            ax.scatter(xs[0:num_compress], np.ones(num_compress)*ts[0], ys[0:num_compress],
#                    marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
#        else:
#            ax.scatter(xs[-num_compress-1:-1], np.ones(num_compress)*ts[-1], ys[-num_compress-1:-1],
#                    marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
#    else:
#        # Plot events
#        ax.scatter(xs, ts, ys,zdir='z', c=colors, facecolors=colors, s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
#
#    ax.view_init(elev=elev, azim=azim)
#    ax.grid(False)
#    # Hide panes
#    ax.xaxis.pane.fill = False
#    ax.yaxis.pane.fill = False
#    ax.zaxis.pane.fill = False
#    if not show_axes:
#        # Hide spines
#        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#        ax.set_frame_on(False)
#    # Hide xy axes
#    ax.set_xticks([])
#    ax.set_yticks([])
#    ax.set_zticks([])
#    # Flush axes
#    ax.set_xlim3d(0, img_size[1])
#    ax.set_ylim3d(ts[0], ts[-1])
#    ax.set_zlim3d(0,img_size[0])
#
#    #ax.xaxis.set_visible(False)
#    #ax.axes.get_yaxis().set_visible(False)
#
#    if show_plot:
#        plt.show()
#    if save_path is not None:
#        ensure_dir(save_path)
#        plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
#    plt.close()
