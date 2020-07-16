import numpy as np
import numpy.lib.recfunctions as nlr
import cv2 as cv
from skimage.measure import block_reduce
import os
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..representations.image import events_to_image
from ..representations.voxel_grid import events_to_voxel
from ..util.event_util import clip_events_to_bounds
from tqdm import tqdm

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

def combine_plotted(root_dir, elev=0, azim=45):
    if elev == 0 and azim == 45:
        pass

def plot_events_sliding(xs, ys, ts, ps, dt=None, sdt=None, save_dir="/tmp", frames=None, frame_ts=None, num_show=1000, event_size=2,
        skip_frames=5, show_skipped=True, elev=0, azim=0, show_events=True, show_frames=True, crop=None,
        compress_front=False, invert=False, num_compress='auto', show_plot=False, show_axes=False):
    if dt is None:
        dt = (ts[-1]-ts[0])/10
        sdt = dt/10
        print("Using dt={}, sdt={}".format(dt, sdt))
    if frames is not None:
        sensor_size = frames[0].shape
    else:
        sensor_size = [max(ys), max(xs)]

    if len(frame_ts.shape) == 2:
        frame_ts = frame_ts[:,1]
    for i, t0 in enumerate(tqdm(np.arange(ts[0], ts[-1]-dt, sdt))):
        te = t0+dt
        eidx0 = np.searchsorted(ts, t0)
        eidx1 = np.searchsorted(ts, te)
        fidx0 = np.searchsorted(frame_ts, t0)
        fidx1 = np.searchsorted(frame_ts, te)
        #print("{}:{} = {}".format(frame_ts[fidx0], ts[eidx0], fidx0))

        wxs, wys, wts, wps = xs[eidx0:eidx1], ys[eidx0:eidx1], ts[eidx0:eidx1], ps[eidx0:eidx1],
        if fidx0 == fidx1:
            wframes=[]
            wframe_ts=[]
        else:
            wframes = frames[fidx0:fidx1]
            wframe_ts = frame_ts[fidx0:fidx1]

        save_path = os.path.join(save_dir, "frame_{:010d}.png".format(i))
        plot_events(wxs, wys, wts, wps, save_path=save_path, num_show=num_show, event_size=event_size,
                imgs=wframes, img_ts=wframe_ts, show_events=show_events, azim=azim,
                elev=elev, show_frames=show_frames, crop=crop, compress_front=compress_front,
                invert=invert, num_compress=num_compress, show_plot=show_plot, img_size=sensor_size,
                show_axes=show_axes)

def plot_voxel_grid(xs, ys, ts, ps, bins=5, frames=[], frame_ts=[],
        sensor_size=None, crop=None, elev=0, azim=45, show_axes=False):
    if sensor_size is None:
        sensor_size = [np.max(ys)+1, np.max(xs)+1] if len(frames)==0 else frames[0].shape
    if crop is not None:
        xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop)
        sensor_size = [crop[2]-crop[0], crop[3]-crop[1]]
        xs, ys = xs-crop[1], ys-crop[0]
    num = 10000
    xs, ys, ts, ps = xs[0:num], ys[0:num], ts[0:num], ps[0:num]
    if len(xs) == 0:
        return
    voxels = events_to_voxel(xs, ys, ts, ps, bins, sensor_size=sensor_size)
    voxels = block_reduce(voxels, block_size=(1,10,10), func=np.mean, cval=0)
    dimdiff = voxels.shape[1]-voxels.shape[0]
    filler = np.zeros((dimdiff, *voxels.shape[1:]))
    voxels = np.concatenate((filler, voxels), axis=0)
    voxels = voxels.transpose(0,2,1)

    pltvoxels = voxels != 0
    pvp, nvp = voxels > 0, voxels < 0
    pvox, nvox = voxels*np.where(voxels > 0, 1, 0), voxels*np.where(voxels < 0, 1, 0)
    pvox, nvox = (pvox/np.max(pvox))*0.5+0.5, (np.abs(nvox)/np.max(np.abs(nvox)))*0.5+0.5
    zeros = np.zeros_like(voxels)

    colors = np.empty(voxels.shape, dtype=object)

    redvals = np.stack((pvox, zeros, pvox-0.5), axis=3)
    redvals = nlr.unstructured_to_structured(redvals).astype('O')

    bluvals = np.stack((nvox-0.5, zeros, nvox), axis=3)
    bluvals = nlr.unstructured_to_structured(bluvals).astype('O')

    colors[pvp] = redvals[pvp]
    colors[nvp] = bluvals[nvp]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(pltvoxels, facecolors=colors, edgecolor='k')
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

    plt.show()

def plot_events(xs, ys, ts, ps, save_path=None, num_compress='auto', num_show=1000,
        event_size=2, elev=0, azim=45, imgs=[], img_ts=[], show_events=True,
        show_frames=True, show_plot=False, crop=None, compress_front=False,
        marker='.', stride = 1, invert=False, img_size=None, show_axes=False):
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
    #Crop events
    if img_size is None:
        img_size = [max(ys), max(ps)] if len(imgs)==0 else imgs[0].shape[0:2]
    crop = [0, 0, img_size[0], img_size[1]] if crop is None else crop
    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
    xs -= crop[1]
    ys -= crop[0]

    #Defaults and range checks
    num_show = len(xs) if num_show == -1 else num_show
    skip = max(len(xs)//num_show, 1)
    num_compress = len(xs) if num_compress == -1 else num_compress
    num_compress = min(img_size[0]*img_size[1]*0.5, len(xs)) if num_compress=='auto' else num_compress
    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]


    ts_scale = 10000

    mlab.figure()

    #Plot images
    if len(imgs)>0 and show_frames:
        for imgidx, (img, img_ts) in enumerate(zip(imgs, img_ts)):
            img = img[crop[0]:crop[2], crop[1]:crop[3]]
            #if len(img.shape)==2:
            #    img = np.stack((img, img, img), axis=2)
            #if num_compress > 0:
            #    events_img = events_to_image(xs[0:num_compress], ys[0:num_compress],
            #            np.ones(num_compress), sensor_size=img.shape[0:2])
            #    events_img[events_img>0] = 1
            #    img[:,:,1]+=events_img[:,:]
            #    img = np.clip(img, 0, 1)
            #x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
            #event_idx = np.searchsorted(ts, img_ts)

            print("img")
            #img = img.transpose(1,0)
            mlab.imshow(img, colormap='gray', extent=[0, img.shape[0], 0, img.shape[1], ts[0]*ts_scale, ts[0]*ts_scale])

           # ax.scatter(xs[0:event_idx], ts[0:event_idx], ys[0:event_idx], zdir='z',
           #         c=colors[0:event_idx], facecolors=colors[0:event_idx],
           #         s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

           # ax.plot_surface(y, img_ts, x, rstride=stride, cstride=stride, facecolors=img, alpha=1 if imgidx == 0 else 0.5)

           # ax.scatter(xs[event_idx:-1], ts[event_idx:-1], ys[event_idx:-1], zdir='z',
           #         c=colors[event_idx:-1], facecolors=colors[event_idx:-1],
           #         s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

    #elif num_compress > 0:
    #    # Plot events
    #    ax.scatter(xs[::skip], ts[::skip], ys[::skip], zdir='z', c=colors[::skip], facecolors=colors[::skip],
    #            s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)
    #    num_compress = min(num_compress, len(xs))
    #    if not compress_front:
    #        ax.scatter(xs[0:num_compress], np.ones(num_compress)*ts[0], ys[0:num_compress],
    #                marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
    #    else:
    #        ax.scatter(xs[-num_compress-1:-1], np.ones(num_compress)*ts[-1], ys[-num_compress-1:-1],
    #                marker=marker, zdir='z', c='w' if invert else 'k', s=np.ones(num_compress)*event_size)
    #else:
    #    # Plot events
    #    ax.scatter(xs, ts, ys,zdir='z', c=colors, facecolors=colors, s=np.ones(xs.shape)*event_size, marker=marker, linewidths=0, alpha=1.0 if show_events else 0)

   # colors = [0 if p>0 else 240 for p in ps[::skip]]
   # N = len(xs[::skip])
   # ones = np.ones(len(xs[::skip]))
   # p3d = mlab.quiver3d(ys[::skip], xs[::skip], ts[::skip]*ts_scale, ones, ones,
   #         ones, scalars=colors, mode='sphere', scale_factor=event_size)
   # p3d.glyph.color_mode = 'color_by_scalar'

   # p3d.module_manager.scalar_lut_manager.lut.table = colors
   # mlab.draw()


    colors = [0 if p>0 else 240 for p in ps[::skip]]
    ones = np.ones(len(xs[::skip]))
    p3d = mlab.quiver3d(ys[::skip], xs[::skip], ts[::skip]*ts_scale, ones, ones,
            ones, scalars=colors, mode='sphere', scale_factor=event_size)
    p3d.glyph.color_mode = 'color_by_scalar'

    p3d.module_manager.scalar_lut_manager.lut.table = colors
    mlab.draw()

    if show_plot:
        mlab.show()
    if save_path is not None:
        ensure_dir(save_path)
        print("Saving to {}".format(save_path))

def plot_between_frames(xs, ys, ts, ps, frames, frame_event_idx, args, plttype='voxel'):
    args.crop = None if args.crop is None else parse_crop(args.crop)
    prev_idx = 0
    for i in range(0, len(frames), args.skip_frames):
        if i != 3:
            continue
        if args.hide_skipped:
            frame = [frames[i]]
            frame_indices = frame_event_idx[i][np.newaxis, ...]
        else:
            frame = frames[i:i+args.skip_frames]
            frame_indices = frame_event_idx[i:i+args.skip_frames]
        print("Processing frame {}".format(i))
        s, e = frame_indices[0,1], frame_indices[-1,0]
        img_ts = []
        for f_idx in frame_indices:
            img_ts.append(ts[f_idx[1]])
        fname = os.path.join(args.output_path, "events_{:09d}.png".format(i))
        if plttype == 'voxel':
            plot_voxel_grid(xs[s:e], ys[s:e], ts[s:e], ps[s:e], bins=args.num_bins, crop=args.crop,
                    frames=frame, frame_ts=img_ts, elev=args.elev, azim=args.azim)
        elif plttype == 'events':
            plot_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], save_path=fname,
                    num_show=args.num_show, event_size=args.event_size, imgs=frame,
                    img_ts=img_ts, show_events=not args.hide_events, azim=args.azim,
                    elev=args.elev, show_frames=not args.hide_frames, crop=args.crop,
                    compress_front=args.compress_front, invert=args.invert,
                    num_compress=args.num_compress, show_plot=args.show_plot, stride=args.stride)
