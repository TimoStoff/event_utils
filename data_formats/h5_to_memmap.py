import argparse
import h5py
import numpy as np
import os, shutil
import json

def find_safe_alternative(output_base_path):
    i = 0
    alternative_path = "{}_{:09d}".format(output_base_path, i)
    while(os.path.exists(alternative_path)):
        i += 1
        alternative_path = "{}_{:09d}".format(output_base_path, i)
        assert(i < 999999999)
    return alternative_path

def save_additional_data_as_mmap(f, mmap_pth, data):
    data_path = os.path.join(mmap_pth, data['mmap_filename'])
    data_ts_path = os.path.join(mmap_pth, data['mmap_ts_filename'])
    data_event_idx_path = os.path.join(mmap_pth, data['mmap_event_idx_filename'])
    data_key = data['h5_key']
    print('Writing {} to mmap {}, timestamps to {}'.format(data_key, data_path, data_ts_path))
    h, w, c = 1, 1, 1
    if data_key in f.keys():
        num_data = len(f[data_key].keys())
        if num_data > 0:
            data_keys = f[data_key].keys()
            data_size = f[data_key][data_keys[0]].attrs['size']
            h, w = data_size[0], data_size[1]
            c = 1 if len(data_size) <= 2 else data_size[2]
    else:
        num_data = 1
    mmp_imgs = np.memmap(data_path, dtype='uint8', mode='w+', shape=(num_data, h, w, c))
    mmp_img_ts = np.memmap(data_ts_path, dtype='float64', mode='w+', shape=(num_data, 1))
    mmp_event_indices = np.memmap(data_event_idx_path, dtype='uint16', mode='w+', shape=(num_data, 1))

    if data_key in f.keys():
        data = []
        data_timestamps = []
        data_event_index = []
        for img_key in f[data_key].keys():
            data.append(f[data_key][img_key][:])
            data_timestamps.append(f[data_key][img_key].attrs['timestamp'])
            data_event_index.append(f[data_key][img_key].attrs['event_idx'])

        data_stack = np.expand_dims(np.stack(data), axis=3)
        data_ts_stack = np.expand_dims(np.stack(data_timestamps), axis=1)
        data_event_indices_stack = np.expand_dims(np.stack(data_event_index), axis=1)
        mmp_imgs[...] = data_stack
        mmp_img_ts[...] = data_ts_stack
        mmp_event_indices[...] = data_event_indices_stack

def write_metadata(f, metadata_path):
    metadata = {}
    for attr in f.attrs:
        val = f.attrs[attr]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        metadata[attr] = val
    with open(metadata_path, 'w') as js:
        json.dump(metadata, js)

def h5_to_memmap(h5_file_path, output_base_path, overwrite=True):
    output_pth = output_base_path
    if os.path.exists(output_pth):
        if overwrite:
            print("Overwriting {}".format(output_pth))
            shutil.rmtree(output_pth)
        else:
            output_pth = find_safe_alternative(output_base_path)
            print('Data will be extracted to: {}'.format(output_pth))
    os.makedirs(output_pth)
    mmap_pth = os.path.join(output_pth, "memmap")
    os.makedirs(mmap_pth)

    ts_path = os.path.join(mmap_pth, 't.npy')
    xy_path = os.path.join(mmap_pth, 'xy.npy')
    ps_path = os.path.join(mmap_pth, 'p.npy')
    metadata_path = os.path.join(mmap_pth, 'metadata.json')

    additional_data = {
            "images":
                {
                    'h5_key' : 'images',
                    'mmap_filename' : 'images.npy',
                    'mmap_ts_filename' : 'timestamps.npy',
                    'mmap_event_idx_filename' : 'image_event_indices.npy',
                    'dims' : 3
                },
            "flow":
                {
                    'h5_key' : 'flow',
                    'mmap_filename' : 'flow.npy',
                    'mmap_ts_filename' : 'flow_timestamps.npy',
                    'mmap_event_idx_filename' : 'flow_event_indices.npy',
                    'dims' : 3
                }
    }

    with h5py.File(h5_file_path, 'r') as f:
        num_events = f.attrs['num_events']
        num_images = f.attrs['num_imgs']
        num_flow = f.attrs['num_flow']

        mmp_ts = np.memmap(ts_path, dtype='float64', mode='w+', shape=(num_events, 1))
        mmp_xy = np.memmap(xy_path, dtype='int16', mode='w+', shape=(num_events, 2))
        mmp_ps = np.memmap(ps_path, dtype='uint8', mode='w+', shape=(num_events, 1))

        mmp_ts[:, 0] = f['events/ts'][:]
        mmp_xy[:, :] = np.stack((f['events/xs'][:], f['events/ys'][:])).transpose()
        mmp_ps[:, 0] = f['events/ps'][:]

        for data in additional_data:
            save_additional_data_as_mmap(f, mmap_pth, additional_data[data])
        write_metadata(f, metadata_path)


if __name__ == "__main__":
    """
    Tool to convert this projects style hdf5 files to the memmap format used in some RPG projects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="HDF5 file to convert")
    parser.add_argument("--output_dir", default=None, help="Path to extract (same as bag if left empty)")
    parser.add_argument('--not_overwrite', action='store_false', help='If set, will not overwrite\
            existing memmap, but will place safe alternative')

    args = parser.parse_args()

    bagname = os.path.splitext(os.path.basename(args.path))[0]
    if args.output_dir is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.path)), bagname)
    else:
        output_path = os.path.join(args.output_dir, bagname)
    h5_to_memmap(args.path, output_path, overwrite=args.not_overwrite)
