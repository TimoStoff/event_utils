import h5py
import numpy as np
import os

def compute_indices(event_stamps, frame_stamps):
    indices_first = np.searchsorted(event_stamps[:,0], frame_stamps[1:])
    indices_last = np.searchsorted(event_stamps[:,0], frame_stamps[:-1])
    index = np.stack([indices_first, indices_last], -1)
    return index

def read_memmap_events(memmap_path, skip_frames=1, return_events=False, images_file = 'images.npy',
        images_ts_file = 'timestamps.npy', optic_flow_file = 'optic_flow.npy',
        optic_flow_ts_file = 'optic_flow_timestamps.npy', events_xy_file = 'xy.npy',
        events_p_file = 'p.npy', events_t_file = 't.npy'):
    assert os.path.isdir(memmap_path), '%s is not a valid memmap_pathectory' % memmap_path

    data = {}
    has_flow = False
    for subroot, _, fnames in sorted(os.walk(memmap_path)):
        for fname in sorted(fnames):
            path = os.path.join(subroot, fname)
            if fname.endswith(".npy"):
                if fname=="index.npy":  # index mapping image index to event idx
                    indices = np.load(path)  # N x 2
                    assert len(indices.shape) == 2 and indices.shape[1] == 2
                    indices = indices.astype("int64")  # ignore event indices which are 0 (before first image)
                    data["index"] = indices.T
                elif fname==images_ts_file:
                    data["frame_stamps"] = np.load(path)[::skip_frames,...]
                elif fname==images_file:
                    data["images"] = np.load(path, mmap_mode="r")[::skip_frames,...]
                elif fname==optic_flow_file:
                    data["optic_flow"] = np.load(path, mmap_mode="r")[::skip_frames,...]
                    has_flow = True
                elif fname==optic_flow_ts_file:
                    data["optic_flow_stamps"] = np.load(path)[::skip_frames,...]

                handle = np.load(path, mmap_mode="r")
                if fname==events_t_file:  # timestamps
                    data["t"] = handle[:].squeeze() if return_events else handle
                    data["t0"] = handle[0]
                elif fname==events_xy_file: # coordinates
                    data["xy"] = handle[:].squeeze() if return_events else handle
                elif fname==events_p_file: # polarity
                    data["p"] = handle[:].squeeze() if return_events else handle

        if len(data) > 0:
            data['path'] = subroot
            if "t" not in data:
                raise Exception(f"Ignoring memmap_pathectory {subroot} since no events")
            if not (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t'])):
                raise Exception(f"Events from {subroot} invalid")
            data["num_events"] = len(data['p'])

            if "index" not in data and "frame_stamps" in data:
                data["index"] = compute_indices(data["t"], data['frame_stamps'])
    return data

def read_memmap_events_dict(memmap_path, skip_frames=1, return_events=False, images_file = 'images.npy',
        images_ts_file = 'timestamps.npy', optic_flow_file = 'optic_flow.npy',
        optic_flow_ts_file = 'optic_flow_timestamps.npy', events_xy_file = 'xy.npy',
        events_p_file = 'p.npy', events_t_file = 't.npy'):
    data = read_memmap_events(memmap_path, skip_frames, return_events, images_file, images_ts_file,
            optic_flow_file, optic_flow_ts_file, events_xy_file, events_p_file, events_t_file)
    events = {
            'xs':data['xy'][:,0].squeeze(),
            'ys':data['xy'][:,1].squeeze(),
            'ts':events['t'][:].squeeze(),
            'ps':events['p'][:].squeeze()}
    return events

def read_h5_events(hdf_path):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = np.stack((f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1)), axis=1)
    else:
        events = np.stack((f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1)), axis=1)
    return events

def read_h5_event_components(hdf_path):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        return (f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1))
    else:
        return (f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1))

def read_h5_events_dict(hdf_path, read_frames=True):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = {
                'xs':f['events/x'][:],
                'ys':f['events/y'][:],
                'ts':f['events/ts'][:],
                'ps':np.where(f['events/p'][:], 1, -1)
        }
        return events
    else:
        events = {
                'xs':f['events/xs'][:],
                'ys':f['events/ys'][:],
                'ts':f['events/ts'][:],
                'ps':np.where(f['events/ps'][:], 1, -1)
                }
        if read_frames:
            images = []
            image_stamps = []
            image_event_indices = []
            for key in f['images']:
                frame = f['images/{}'.format(key)][:]
                images.append(frame)
                image_stamps.append(f['images/{}'.format(key)].attrs['timestamp'])
                image_event_indices.append(f['images/{}'.format(key)].attrs['event_idx'])
            events['frames'] = images
            #np.concatenate(images, axis=2).swapaxes(0,2) if len(frame.shape)==3 else np.stack(images, axis=0)
            events['frame_timestamps'] = np.array(image_stamps)
            events['frame_event_indices'] = np.array(image_event_indices)
        return events
