from mayavi import mlab
from mayavi.api import Engine
import numpy as np
import numpy.lib.recfunctions as nlr
import cv2 as cv
from skimage.measure import block_reduce
import os
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from ..representations.image import events_to_image
from ..representations.voxel_grid import events_to_voxel
from ..util.event_util import clip_events_to_bounds
from ..visualization.visualization_utils import *
from tqdm import tqdm

def plot_events_sliding(xs, ys, ts, ps, args, dt=None, sdt=None, frames=None, frame_ts=None, padding=True):

    skip = max(len(xs)//args.num_show, 1)
    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]
    t0 = ts[0]
    sx,sy, st, sp = [], [], [], []
    if padding:
        for i in np.arange(ts[0]-dt, ts[0], sdt):
            sx.append(0)
            sy.append(0)
            st.append(i)
            sp.append(0)
        print(len(sx))
        print(st)
        print(ts)
        xs = np.concatenate((np.array(sx), xs))
        ys = np.concatenate((np.array(sy), ys))
        ts = np.concatenate((np.array(st), ts))
        ps = np.concatenate((np.array(sp), ps))
        print(ts)

        ts += -st[0]
        frame_ts += -st[0]
        t0 += -st[0]
        print(ts)

    f = mlab.figure(bgcolor=(1,1,1), size=(1080, 720))
    engine = mlab.get_engine()
    scene = engine.scenes[0]
    scene.scene.camera.position = [373.1207907160101, 5353.96218497846, 7350.065665045519]
    scene.scene.camera.focal_point = [228.0033999234376, 37.75424682790012, 3421.439332472788]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.9997493712140433, -0.02027499237784438, -0.009493125997461629]
    scene.scene.camera.clipping_range = [2400.251302762254, 11907.415293888362]
    scene.scene.camera.compute_view_plane_normal()

    print("ts from {} to {}, imgs from {} to {}".format(ts[0], ts[-1], frame_ts[0], frame_ts[-1]))
    frame_ts = np.array([t0]+list(frame_ts[0:-1]))
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

        save_path = os.path.join(args.output_path, "frame_{:010d}.jpg".format(i))
        plot_events(wxs, wys, wts, wps, save_path=save_path, num_show=-1, event_size=args.event_size,
                imgs=wframes, img_ts=wframe_ts, show_events=not args.hide_events, azim=args.azim,
                elev=args.elev, show_frames=not args.hide_frames, crop=args.crop, compress_front=args.compress_front,
                invert=args.invert, num_compress=args.num_compress, show_plot=args.show_plot, img_size=sensor_size,
                show_axes=args.show_axes, ts_scale=args.ts_scale)

        if save_path is not None:
            ensure_dir(save_path)
            #mlab.savefig(save_path, figure=f, magnification=10)
            #GUI().process_events()
            #img = mlab.screenshot(figure=f, mode='rgba', antialiased=True)
            #print(img.shape)
            mlab.savefig(save_path, figure=f, magnification=8)

        mlab.clf()

def plot_voxel_grid(xs, ys, ts, ps, bins=5, frames=[], frame_ts=[],
        sensor_size=None, crop=None, elev=0, azim=45, show_axes=False):
    if sensor_size is None:
        sensor_size = [np.max(ys)+1, np.max(xs)+1] if len(frames)==0 else frames[0].shape
    if crop is not None:
        xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop)
        sensor_size = crop_to_size(crop)
        xs, ys = xs-crop[2], ys-crop[0]
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
        marker='.', stride = 1, invert=False, img_size=None, show_axes=False,
        ts_scale = 100000):
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
    print("plot all")
    #Crop events
    if img_size is None:
        img_size = [max(ys), max(ps)] if len(imgs)==0 else imgs[0].shape[0:2]
    crop = [0, img_size[0], 0, img_size[1]] if crop is None else crop
    xs, ys, ts, ps = clip_events_to_bounds(xs, ys, ts, ps, crop, set_zero=False)
    xs, ys = xs-crop[2], ys-crop[0]

    #Defaults and range checks
    num_show = len(xs) if num_show == -1 else num_show
    skip = max(len(xs)//num_show, 1)
    print("Has {} events, show only {}, skip = {}".format(len(xs), num_show, skip))
    num_compress = len(xs) if num_compress == -1 else num_compress
    num_compress = min(img_size[0]*img_size[1]*0.5, len(xs)) if num_compress=='auto' else num_compress
    xs, ys, ts, ps = xs[::skip], ys[::skip], ts[::skip], ps[::skip]

    t0 = ts[0]
    ts = ts-t0

    #mlab.options.offscreen = True

    #Plot images
    if len(imgs)>0 and show_frames:
        for imgidx, (img, img_t) in enumerate(zip(imgs, img_ts)):
            img = img[crop[0]:crop[1], crop[2]:crop[3]]

            mlab.imshow(img, colormap='gray', extent=[0, img.shape[0], 0, img.shape[1], (img_t-t0)*ts_scale, (img_t-t0)*ts_scale+0.01], opacity=1.0, transparent=False)

    colors = [0 if p>0 else 240 for p in ps]
    ones = np.array([0 if p==0 else 1 for p in ps])
    p3d = mlab.quiver3d(ys, xs, ts*ts_scale, ones, ones, ones, scalars=colors, mode='sphere', scale_factor=event_size)
    p3d.glyph.color_mode = 'color_by_scalar'
    p3d.module_manager.scalar_lut_manager.lut.table = colors
    #mlab.draw()

    #mlab.view(84.5, 54, 5400, np.array([ 187,  175, 2276]), roll=95)

    if show_plot:
        mlab.show()
    #if save_path is not None:
    #    ensure_dir(save_path)
    #    print("Saving to {}".format(save_path))
    #    imgmap = mlab.screenshot(mode='rgba', antialiased=True)
    #    print(imgmap.shape)
    #    cv.imwrite(save_path, imgmap)

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
            print("plot events")
            plot_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], save_path=fname,
                    num_show=args.num_show, event_size=args.event_size, imgs=frame,
                    img_ts=img_ts, show_events=not args.hide_events, azim=args.azim,
                    elev=args.elev, show_frames=not args.hide_frames, crop=args.crop,
                    compress_front=args.compress_front, invert=args.invert,
                    num_compress=args.num_compress, show_plot=args.show_plot, stride=args.stride)
