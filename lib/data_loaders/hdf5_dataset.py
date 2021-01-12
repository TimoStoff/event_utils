import h5py
from ..util.event_util import binary_search_h5_dset
from .base_dataset import BaseVoxelDataset
import matplotlib.pyplot as plt

class DynamicH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path):
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = self.data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    args = parser.parse_args()

    dloader = DynamicH5Dataset(args.path)
    for item in dloader:
        print(item['events'].shape)
