import argparse
import os
import numpy as np
from lib.data_formats.read_events import read_memmap_events, read_h5_events_dict

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
    parser.add_argument('--w_width', type=float, default=0.01,
                        help='new plot is formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sw_width', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')

    parser.add_argument("--num_bins", type=int, default=6, help="How many bins voxels should have.")

    parser.add_argument('--show_plot', action='store_true', help='If true, will also display the plot in an interactive window.\
            Useful for selecting the desired orientation.')

    parser.add_argument("--num_show", type=int, default=-1, help="How many events to show per plot. If -1, show all events.")
    parser.add_argument("--event_size", type=float, default=2, help="Marker size of the plotted events")
    parser.add_argument("--ts_scale", type=int, default=10000, help="Scales the time axis. Only applicable for mayavi rendering.")
    parser.add_argument("--elev", type=float, default=20, help="Elevation of plot")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth of plot")
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
    parser.add_argument("--renderer", type=str, default="matplotlib", help="Which renderer to use (mayavi is faster)", choices=["matplotlib", "mayavi"])
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
        frames = [np.flip(np.flip(x/255., axis=0), axis=1) for x in events['frames']]
        frame_ts = events['frame_timestamps'][1:]-t0
        frame_end = events['frame_event_indices'][1:]
        frame_start = np.concatenate((np.array([0]), frame_end))
        frame_idx = np.stack((frame_end, frame_start[0:-1]), axis=1)
        ys = frames[0].shape[0]-ys
        xs = frames[0].shape[1]-xs

    if args.plot_method == 'between_frames':
        if args.renderer == "mayavi":
            from lib.visualization.draw_event_stream_mayavi import plot_between_frames
            plot_between_frames(xs, ys, ts, ps, frames, frame_idx, args, plttype='events')
        elif args.renderer == "matplotlib":
            from lib.visualization.draw_event_stream import plot_between_frames
            plot_between_frames(xs, ys, ts, ps, frames, frame_idx, args, plttype='events')
    elif args.plot_method == 'k_events':
        print(args.renderer)
        pass
    elif args.plot_method == 't_seconds':
        if args.renderer == "mayavi":
            from lib.visualization.draw_event_stream_mayavi import plot_events_sliding
            plot_events_sliding(xs, ys, ts, ps, args, dt=args.w_width, sdt=args.sw_width, frames=frames, frame_ts=frame_ts)
        elif args.renderer == "matplotlib":
            from lib.visualization.draw_event_stream import plot_events_sliding
            plot_events_sliding(xs, ys, ts, ps, args, frames=frames, frame_ts=frame_ts)
