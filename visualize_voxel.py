import argparse
import os
import numpy as np
from lib.visualization.draw_event_stream import plot_between_frames
from lib.data_formats.read_events import read_memmap_events, read_h5_events_dict

def plot_events_sliding(xs, ys, ts, ps, args, frames=None, frame_ts=None):
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

        save_path = os.path.join(args.output_path, "frame_{:010d}.png".format(i))
        plot_events(wxs, wys, wts, wps, save_path=save_path, num_show=args.num_show, event_size=args.event_size,
                imgs=args.wframes, img_ts=args.wframe_ts, show_events=args.show_events, azim=args.azim,
                elev=args.elev, show_frames=args.show_frames, crop=args.crop, compress_front=args.compress_front,
                invert=args.invert, num_compress=args.num_compress, show_plot=args.show_plot, img_size=args.sensor_size,
                show_axes=args.show_axes)

if __name__ == "__main__":
    """
    Quick demo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="memmap events path")
    parser.add_argument("--output_path", type=str, default="/tmp/visualization", help="Where to save image outputs")

    parser.add_argument('--plot_method', default='between_frames', type=str,
                        help='which method should be used to visualize',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new plot is formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new plot is formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument("--num_bins", type=int, default=6, help="How many bins voxels should have.")

    parser.add_argument('--show_plot', action='store_true', help='If true, will also display the plot in an interactive window.\
            Useful for selecting the desired orientation.')

    parser.add_argument("--num_show", type=int, default=-1, help="How many events to show per plot. If -1, show all events.")
    parser.add_argument("--event_size", type=float, default=2, help="Marker size of the plotted events")
    parser.add_argument("--elev", type=float, default=20, help="Elevation of plot")
    parser.add_argument("--azim", type=float, default=-25, help="Azimuth of plot")
    parser.add_argument("--stride", type=int, default=1, help="Downsample stride for plotted images.")
    parser.add_argument("--skip_frames", type=int, default=1, help="Amount of frames to place per plot.")
    parser.add_argument("--start_frame", type=int, default=0, help="On which frame to start.")
    parser.add_argument('--hide_skipped', action='store_true', help='Do not draw skipped frames into plot.')
    parser.add_argument('--hide_events', action='store_true', help='Do not draw events')
    parser.add_argument('--hide_frames', action='store_true', help='Do not draw frames')
    parser.add_argument('--show_axes', action='store_true', help='Draw axes')
    parser.add_argument("--num_compress", type=int, default=0, help="How many events to draw compressed. If 'auto'\
            will automatically determine.", choices=['value', 'auto'])
    parser.add_argument('--compress_front', action='store_true', help='If set, will put the compressed events at the _start_\
            of the event volume, rather than the back.')
    parser.add_argument('--invert', action='store_true', help='If the figure is for a black background, you can invert the \
            colors for better visibility.')
    parser.add_argument("--crop", type=str, default=None, help="Set a crop of both images and events. Uses 'imagemagick' \
            syntax, eg for a crop of 10x20 starting from point 30,40 use: 10x20+30+40.")
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
        frames = [np.flip(x/255., axis=0) for x in events['frames']]
        frame_ts = events['frame_timestamps'][1:]-t0
        frame_end = events['frame_event_indices'][1:]
        frame_start = np.concatenate((np.array([0]), frame_end))
        frame_idx = np.stack((frame_end, frame_start[0:-1]), axis=1)
        ys = frames[0].shape[0]-ys

    plot_between_frames(xs, ys, ts, ps, frames, frame_idx, args, plttype='voxel')
