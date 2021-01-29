import argparse
import os
from tqdm import tqdm
import numpy as np
from lib.data_formats.read_events import read_memmap_events, read_h5_events_dict
from lib.data_loaders import MemMapDataset, DynamicH5Dataset, NpyDataset
from lib.visualization.visualizers import TimeStampImageVisualizer, EventImageVisualizer, \
        EventsVisualizer, VoxelVisualizer

if __name__ == "__main__":
    """
    Quick demo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="memmap events path")
    parser.add_argument("--output_path", type=str, default="/tmp/visualization", help="Where to save image outputs")
    parser.add_argument("--filetype", type=str, default="png", help="Which filetype to save as", choices=["png", "jpg", "pdf"])

    parser.add_argument('--plot_method', default='between_frames', type=str,
                        help='which method should be used to visualize',
                        choices=['between_frames', 'k_events', 't_seconds', 'fixed_frames'])
    parser.add_argument('--w_width', type=float, default=0.01,
                        help='new plot is formed every t seconds/k events (required if voxel_method is t_seconds)')
    parser.add_argument('--sw_width', type=float,
                        help='sliding_window size in seconds/events (required if voxel_method is t_seconds)')
    parser.add_argument('--num_frames', type=int, default=100, help='if fixed_frames chosen as voxel method, sets the number of frames')

    parser.add_argument('--visualization', type=str, default='events', choices=['events', 'voxels', 'event_image', 'ts_image'])

    parser.add_argument("--num_bins", type=int, default=6, help="How many bins voxels should have.")

    parser.add_argument('--show_plot', action='store_true', help='If true, will also display the plot in an interactive window.\
            Useful for selecting the desired orientation.')

    parser.add_argument("--num_show", type=int, default=-1, help="How many events to show per plot. If -1, show all events.")
    parser.add_argument("--event_size", type=float, default=2, help="Marker size of the plotted events")
    parser.add_argument("--ts_scale", type=int, default=10000, help="Scales the time axis. Only applicable for mayavi rendering.")
    parser.add_argument("--elev", type=float, default=0, help="Elevation of plot")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth of plot")
    parser.add_argument("--stride", type=int, default=1, help="Downsample stride for plotted images.")
    parser.add_argument("--skip_frames", type=int, default=1, help="Amount of frames to place per plot.")
    parser.add_argument("--start_frame", type=int, default=0, help="On which frame to start.")
    parser.add_argument('--hide_skipped', action='store_true', help='Do not draw skipped frames into plot.')
    parser.add_argument('--hide_events', action='store_true', help='Do not draw events')
    parser.add_argument('--hide_frames', action='store_true', help='Do not draw frames')
    parser.add_argument('--show_axes', action='store_true', help='Draw axes')
    parser.add_argument('--flip_x', action='store_true', help='Flip in the x axis')
    parser.add_argument("--num_compress", type=str, default='auto', help="How many events to draw compressed. If 'auto'\
            will automatically determine.", choices=['auto', 'none', 'all'])
    parser.add_argument('--compress_front', action='store_true', help='If set, will put the compressed events at the _start_\
            of the event volume, rather than the back.')
    parser.add_argument('--invert', action='store_true', help='If the figure is for a black background, you can invert the \
            colors for better visibility.')
    parser.add_argument("--crop", type=str, default=None, help="Set a crop of both images and events. Uses 'imagemagick' \
            syntax, eg for a crop of 10x20 starting from point 30,40 use: 10x20+30+40.")
    parser.add_argument("--renderer", type=str, default="matplotlib", help="Which renderer to use (mayavi is faster)", choices=["matplotlib", "mayavi"])
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if os.path.isdir(args.path):
        loader_type = MemMapDataset
    elif os.path.splitext(args.path)[1] == ".npy":
        loader_type = NpyDataset
    else:
        loader_type = DynamicH5Dataset
    dataloader = loader_type(args.path, voxel_method={'method':args.plot_method, 't':args.w_width,
        'k':args.w_width, 'sliding_window_t':args.sw_width, 'sliding_window_w':args.sw_width, 'num_frames':args.num_frames},
            return_events=True, return_voxelgrid=False, return_frame=True, return_flow=True, return_format='numpy')
    sensor_size = dataloader.size()

    if args.visualization == 'events':
        kwargs = {'num_compress':args.num_compress, 'num_show':args.num_show, 'event_size':args.event_size,
                'elev':args.elev, 'azim':args.azim, 'show_events':not args.hide_events,
                'show_frames':not args.hide_frames, 'show_plot':args.show_plot, 'crop':args.crop,
                'compress_front':args.compress_front, 'marker':'.', 'stride':args.stride,
                'invert':args.invert, 'show_axes':args.show_axes, 'flip_x':args.flip_x}
        visualizer = EventsVisualizer(sensor_size)
    elif args.visualization == 'voxels':
        kwargs = {'bins':args.num_bins, 'crop':args.crop, 'elev':args.elev, 'azim':args.azim,
                'show_axes':args.show_axes, 'show_plot':args.show_plot, 'flip_x':args.flip_x}
        visualizer = VoxelVisualizer(sensor_size)
    elif args.visualization == 'event_image':
        kwargs = {}
        visualizer = EventImageVisualizer(sensor_size)
    elif args.visualization == 'ts_image':
        kwargs = {}
        visualizer = TimeStampImageVisualizer(sensor_size)
    else:
        raise Exception("Unknown visualization chosen: {}".format(args.visualization))

    plot_data = {'events':np.ones((0, 4)), 'frame':[], 'frame_ts':[]}
    print("{} frames in sequence".format(len(dataloader)))
    for i, data in enumerate(tqdm(dataloader)):
        plot_data['events'] = np.concatenate((plot_data['events'], data['events']))
        if args.plot_method == 'between_frames':
            plot_data['frame'].append(data['frame'])
            plot_data['frame_ts'].append(data['frame_ts'])
        else:
            plot_data['frame'] = data['frame']
            plot_data['frame_ts'] = data['frame_ts']

        output_path = os.path.join(args.output_path, "frame_{:010d}.{}".format(i, args.filetype))
        if i%args.skip_frames == 0:
            visualizer.plot_events(plot_data, output_path, **kwargs)
            plot_data = {'events':np.ones((0, 4)), 'frame':[], 'frame_ts':[]}

    #if args.plot_method == 'between_frames':
    #    if args.renderer == "mayavi":
    #        from lib.visualization.draw_event_stream_mayavi import plot_between_frames
    #        plot_between_frames(xs, ys, ts, ps, frames, frame_idx, args, plttype='events')
    #    elif args.renderer == "matplotlib":
    #        from lib.visualization.draw_event_stream import plot_between_frames
    #        plot_between_frames(xs, ys, ts, ps, frames, frame_idx, args, plttype='events')
    #elif args.plot_method == 'k_events':
    #    print(args.renderer)
    #    pass
    #elif args.plot_method == 't_seconds':
    #    if args.renderer == "mayavi":
    #        from lib.visualization.draw_event_stream_mayavi import plot_events_sliding
    #        plot_events_sliding(xs, ys, ts, ps, args, dt=args.w_width, sdt=args.sw_width, frames=frames, frame_ts=frame_ts)
    #    elif args.renderer == "matplotlib":
    #        from lib.visualization.draw_event_stream import plot_events_sliding
    #        plot_events_sliding(xs, ys, ts, ps, args, frames=frames, frame_ts=frame_ts)
