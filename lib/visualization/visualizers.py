import numpy as np
import numpy.lib.recfunctions as nlr
import cv2 as cv
import colorsys
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

    def plot_events(self, data, save_path, **kwargs):
        raise NotImplementedError

    @staticmethod
    def unpackage_events(events):
        return events[:,0].astype(int), events[:,1].astype(int), events[:,2], events[:,3]

class TimeStampImageVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.ts_img = TimestampImage(sensor_size)
        self.sensor_size = sensor_size

    def plot_events(self, data, save_path, **kwargs):
        xs, ys, ts, ps = self.unpackage_events(data['events'])
        self.ts_img.set_init(ts[0])
        self.ts_img.add_events(xs, ys, ts, ps)
        timestamp_image = self.ts_img.get_image()
        fig = plt.figure()
        plt.imshow(timestamp_image, cmap='viridis')
        ensure_dir(save_path)
        plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
        #plt.show()

class EventImageVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

    def plot_events(self, data, save_path, **kwargs):
        xs, ys, ts, ps = self.unpackage_events(data['events'])
        img = events_to_image(xs.astype(int), ys.astype(int), ps, self.sensor_size, interpolation=None, padding=False)
        mn, mx = np.min(img), np.max(img)
        img = (img-mn)/(mx-mn)

        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        ensure_dir(save_path)
        plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
        #plt.show()


class EventsVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

    def plot_events(self, data, save_path,
            num_compress='auto', num_show=1000,
            event_size=2, elev=0, azim=45, show_events=True,
            show_frames=True, show_plot=False, crop=None, compress_front=False,
            marker='.', stride = 1, invert=False, show_axes=False, flip_x=False):
        """
        Given events, plot these in a spatiotemporal volume.
        :param: xs x coords of events
        :param: ys y coords of events
        :param: ts t coords of events
        :param: ps p coords of events
        :param: save_path if set, will save plot to here
        :param: num_compress will take this number of events from the end
            and create an event image from these. This event image will
            be displayed at the end of the spatiotemporal volume
        :param: num_show sets the number of events to plot. If set to -1
            will plot all of the events (can be potentially expensive)
        :param: event_size sets the size of the plotted events
        :param: elev sets the elevation of the plot
        :param: azim sets the azimuth of the plot
        :param: imgs a list of images to draw into the spatiotemporal volume
        :param: img_ts a list of the position on the temporal axis where each
            image from 'imgs' is to be placed (the timestamp of the images, usually)
        :param: show_events if False, will not plot the events (only images)
        :param: crop a list of length 4 that sets the crop of the plot (must
            be in the format [top_left_y, top_left_x, height, width]
        """
        xs, ys, ts, ps = self.unpackage_events(data['events'])
        imgs, img_ts = data['frame'], data['frame_ts']
        if not (isinstance(imgs, list) or isinstance(imgs, tuple)):
            imgs, img_ts = [imgs], [img_ts]

        ys = self.sensor_size[0]-ys
        xs = self.sensor_size[1]-xs if flip_x else xs
        #Crop events
        img_size = self.sensor_size
        if img_size is None:
            img_size = [max(ys), max(ps)] if len(imgs)==0 else imgs[0].shape[0:2]
        crop = [0, img_size[0], 0, img_size[1]] if crop is None else crop
        xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
        xs, ys = xs-crop[2], ys-crop[0]

        if len(xs) < 2:
            xs = np.array([0,0])
            ys = np.array([0,0])
            if img_ts is None:
                ts = np.array([0,0])
            else:
                ts = np.array([img_ts[0], img_ts[0]+0.000001])
            ps = np.array([0.,0.])

        #Defaults and range checks
        num_show = len(xs) if num_show == -1 else num_show
        skip = max(len(xs)//num_show, 1)
        num_compress = len(xs) if num_compress == 'all' else num_compress
        num_compress = min(int(img_size[0]*img_size[1]*0.5), len(xs)) if num_compress=='auto' else 0
        xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]

        #Prepare the plot, set colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
        colors = ['r' if p>0 else ('#00DAFF' if invert else 'b') for p in ps]

        #Plot images
        if len(imgs)>0 and show_frames:
            for imgidx, (img, img_ts) in enumerate(zip(imgs, img_ts)):
                img = img[crop[0]:crop[1], crop[2]:crop[3]].astype(float)
                img = np.flip(img, axis=0)
                img = np.flip(img, axis=1) if flip_x else img
                if len(img.shape)==2:
                    img = np.stack((img, img, img), axis=2)
                if num_compress > 0:
                    events_img = events_to_image(xs[0:num_compress], ys[0:num_compress],
                            np.ones(min(num_compress, len(xs))), sensor_size=img.shape[0:2])
                    events_img[events_img>0] = 1
                    img[:,:,1] += events_img[:,:]
                    img = np.clip(img, 0, 1)
                x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
                event_idx = np.searchsorted(ts, img_ts)

                ax.scatter(xs[0:event_idx], ts[0:event_idx], ys[0:event_idx], zdir='z',
                        c=colors[0:event_idx], facecolors=colors[0:event_idx],
                        s=event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

                img /= 255.0
                #img = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX)
                ax.plot_surface(y, img_ts, x, rstride=stride, cstride=stride, facecolors=img, alpha=1)

                ax.scatter(xs[event_idx:-1], ts[event_idx:-1], ys[event_idx:-1], zdir='z',
                        c=colors[event_idx:-1], facecolors=colors[event_idx:-1],
                        s=event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
    
        elif num_compress > 0:
            # Plot events
            ax.scatter(xs[::skip], ts[::skip], ys[::skip], zdir='z', c=colors[::skip], facecolors=colors[::skip],
                    s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
            num_compress = min(num_compress, len(xs))
            if not compress_front:
                ax.scatter(xs[0:num_compress], np.ones(num_compress)*ts[0], ys[0:num_compress],
                        marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
            else:
                ax.scatter(xs[-num_compress-1:-1], np.ones(num_compress)*ts[-1], ys[-num_compress-1:-1],
                        marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
        else:
            # Plot events
            ax.scatter(xs, ts, ys,zdir='z', c=colors, facecolors=colors, s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
    
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        # Hide panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if not show_axes:
            # Hide spines
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.set_frame_on(False)
        # Hide xy axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Flush axes
        ax.set_xlim3d(0, img_size[1])
        ax.set_ylim3d(ts[0], ts[-1])
        ax.set_zlim3d(0,img_size[0])
        #ax.xaxis.set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)

        if show_plot:
            plt.show()
        if save_path is not None:
            ensure_dir(save_path)
            print("Saving to {}".format(save_path))
            plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
        plt.close()

class VoxelVisualizer(Visualizer):

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size

    @staticmethod
    def increase_brightness(rgb, increase=0.5):
        rgb = (rgb*255).astype('uint8')
        channels = rgb.shape[1]
        hsv = (np.stack([cv.cvtColor(rgb[:,x,:,:], cv.COLOR_RGB2HSV) for x in range(channels)])).astype(float)
        hsv[:,:,:,2] = np.clip(hsv[:,:,:,2] + increase*255, 0, 255)
        hsv = hsv.astype('uint8')
        rgb_new = np.stack([cv.cvtColor(hsv[x,:,:,:], cv.COLOR_HSV2RGB) for x in range(channels)])
        rgb_new = (rgb_new.transpose(1,0,2,3)).astype(float)
        return rgb_new/255.0

    def plot_events(self, data, save_path, bins=5, crop=None, elev=0, azim=45, show_axes=False,
            show_plot=False, flip_x=False, size_reduction=10):

        xs, ys, ts, ps = self.unpackage_events(data['events'])
        if len(xs) < 2:
            return
        ys = self.sensor_size[0]-ys
        xs = self.sensor_size[1]-xs if flip_x else xs

        frames, frame_ts = data['frame'], data['frame_ts']
        if not isinstance(frames, list):
            frames, frame_ts = [frames], [frame_ts]

        if self.sensor_size is None:
            self.sensor_size = [np.max(ys)+1, np.max(xs)+1] if len(frames)==0 else frames[0].shape
        if crop is not None:
            xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop)
            self.sensor_size = crop_to_size(crop)
            xs, ys = xs-crop[2], ys-crop[0]
        num = 10000
        xs, ys, ts, ps = xs[0:num], ys[0:num], ts[0:num], ps[0:num]
        if len(xs) == 0:
            return
        voxels = events_to_voxel(xs, ys, ts, ps, bins, sensor_size=self.sensor_size)
        voxels = block_reduce(voxels, block_size=(1,size_reduction,size_reduction), func=np.mean, cval=0)
        dimdiff = voxels.shape[1]-voxels.shape[0]
        filler = np.zeros((dimdiff, *voxels.shape[1:]))
        voxels = np.concatenate((filler, voxels), axis=0)
        voxels = voxels.transpose(0,2,1)

        pltvoxels = voxels != 0
        pvp, nvp = voxels > 0, voxels < 0
        rng = 0.2
        min_r, min_b, max_g = 80/255.0, 80/255.0, 0/255.0

        vox_cols = voxels/(max(np.abs(np.max(voxels)), np.abs(np.min(voxels))))
        pvox, nvox = vox_cols*np.where(vox_cols > 0, 1, 0), np.abs(vox_cols)*np.where(vox_cols < 0, 1, 0)
        pvox, nvox = pvox*(1-min_r)+min_r, nvox*(1-min_b)+min_b
        zeros = np.zeros_like(voxels)

        colors = np.empty(voxels.shape, dtype=object)

        increase = 0.5
        redvals = np.stack((pvox, (1.0-pvox)*max_g, pvox-min_r), axis=3)
        redvals = self.increase_brightness(redvals, increase=increase)
        redvals = nlr.unstructured_to_structured(redvals).astype('O')

        bluvals = np.stack((nvox-min_b, (1.0-nvox)*max_g, nvox), axis=3)
        bluvals = self.increase_brightness(bluvals, increase=increase)
        bluvals = nlr.unstructured_to_structured(bluvals).astype('O')

        colors[pvp] = redvals[pvp]
        colors[nvp] = bluvals[nvp]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(pltvoxels, facecolors=colors)
        ax.view_init(elev=elev, azim=azim)

        ax.grid(False)
        # Hide panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if not show_axes:
            # Hide spines
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.set_frame_on(False)
        # Hide xy axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if show_plot:
            plt.show()
        if save_path is not None:
            ensure_dir(save_path)
            print("Saving to {}".format(save_path))
            plt.savefig(save_path, transparent=True, dpi=600, bbox_inches = 'tight')
        plt.close()
