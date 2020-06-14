import argparse
import numpy as np
import os
from data_formats.read_events import read_memmap_events, read_h5_events_dict
from representations.image import events_to_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating {directory}")
        os.makedirs(directory)

def combine_plotted(root_dir, elev=0, azim=45):
    if elev == 0 and azim == 45:
        pass

def plot_events(xs, ys, ts, ps, save_path=None, num_compress=0, num_show=1000,
        size=1, elev=0, azim=45, imgs=[], img_ts=[], show_events=True):
    num_show = len(xs) if num_show == -1 else num_show
    skip = max(len(xs)//num_show, 1)
    num_compress = len(xs) if num_compress == -1 else num_compress
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    colors = ['r' if p>0 else 'b' for p in ps]
    # Plot events
    ax.scatter(xs[::skip], ts[::skip], ys[::skip], zdir='z', c=colors[::skip],
            s=np.ones(xs.shape)*size, alpha=1 if show_events else 0)
    if len(imgs)>0:
        for img, img_ts in zip(imgs, img_ts):
            if num_compress > 0:
                events_img = events_to_image(xs[0:num_compress], ys[0:num_compress],
                        np.ones(num_compress), sensor_size=img.shape[0:2])
                events_img[events_img>0] = 1
                img[:,:,1]+=events_img[:,:]
                img = np.clip(img, 0, 1)
            x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
            stride = 4
            ax.plot_surface(y, img_ts, x, rstride=stride, cstride=stride, facecolors=img)
    else:
        num_compress = min(num_compress, len(xs))
        ax.scatter(xs[0:num_compress], np.ones(num_compress)*ts[0], ys[0:num_compress], zdir='z', c='k', s=np.ones(num_compress)*size)


    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    # Hide panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
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
    #ax.set_xlim3d(0, 1000)
    ax.set_ylim3d(ts[0],ts[-1])
    #ax.set_zlim3d(0,1000)

    ax.xaxis.set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if show_events:
        plt.show()
    else:
        if save_path is None:
            raise Exception("No save path given")
        ensure_dir(save_path)
        plt.savefig(save_path, transparent=True)
    plt.close()

def plot_events_between_frames(xs, ys, ts, ps, frames, frame_event_idx, save_dir, num_show=1000,
        skip_frames=5, show_skipped=True, elev=0, azim=0, show_events=True):
    prev_idx = 0
    for i in range(0, len(frames), skip_frames):
        if show_skipped:
            frame = frames[i:i+skip_frames]
            frame_indices = frame_event_idx[i:i+skip_frames]
        else:
            frame = [frames[i]]
            frame_indices = frame_event_idx[i][np.newaxis, ...]
        print("Processing frame {}".format(i))
        s, e = frame_indices[0,1], frame_indices[-1,0]
        img_ts = []
        for f_idx in frame_indices:
            img_ts.append(ts[f_idx[1]])
        fname = os.path.join(save_dir, "events_{:09d}.png".format(i))
        plot_events(xs[s:e], ys[s:e], ts[s:e], ps[s:e], save_path=fname, num_show=num_show,
                imgs=frame, img_ts=img_ts, num_compress=0, show_events=show_events, azim=azim, elev=elev)

if __name__ == "__main__":
    """
    Quick demo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="memmap events path")
    parser.add_argument("--output_path", type=str, default="/tmp/visualization", help="Where to save image outputs")
    parser.add_argument("--num_show", type=int, default=1000, help="How many events to show per plot. If -1, show all events.")
    parser.add_argument("--elev", type=float, default=0, help="Elevation of plot")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth of plot")
    parser.add_argument("--skip_frames", type=int, default=1, help="Amount of frames to place per plot.")
    parser.add_argument("--start_frame", type=int, default=0, help="On which frame to start.")
    parser.add_argument('--hide_skipped', action='store_true', help='Do not draw skipped frames into plot.')
    parser.add_argument('--hide_events', action='store_true', help='Do not draw events (frames only).')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        events = read_memmap_events(args.path)

        ts = events['t'][:].squeeze()
        t0 = ts[0]
        ts = ts-t0
        frames = (events['images'][args.start_frame+1::])/255
        frame_idx = events['index'][args.start_frame::]
        frame_ts = events['frame_stamps'][args.start_frame+1::]-t0

        start_idx = np.searchsorted(ts, frame_ts[0])
        print("Starting from frame {}, event {}".format(args.start_frame, start_idx))

        xs = events['xy'][:,0]
        ys = events['xy'][:,1]
        ts = ts[:]
        ps = events['p'][:]

        print("Have {} frames".format(frames.shape))
    else:
        events = read_h5_events_dict(args.path)
        xs = events['xs']
        ys = events['ys']
        ts = events['ts']
        ps = events['ps']
        t0 = ts[0]
        ts = ts-t0
        frames = events['frames']
        frame_ts = events['frame_timestamps']-t0
        frame_end = events['frame_event_indices']
        frame_start = np.concatenate((np.array([0]), frame_end))
        frame_idx = np.stack((frame_end, frame_start[0:-1]), axis=1)

    num_to_plot = args.num_show if args.num_show >= 0 else len(xs)
    plot_events_between_frames(xs, ys, ts, ps, frames, frame_idx, args.output_path, num_show=args.num_show,
            skip_frames=args.skip_frames, show_skipped=not args.hide_skipped, azim=args.azim, elev=args.elev,
            show_events=not args.hide_events)
