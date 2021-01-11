from .base_dataset import BaseVoxelDataset
import numpy as np

class NpyDataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        return None

    def get_flow(self, index):
        return None

    def get_events(self, idx0, idx1):
        xs = self.xs[idx0:idx1]
        ys = self.ys[idx0:idx1]
        ts = self.ts[idx0:idx1]
        ps = self.ps[idx0:idx1]
        return xs, ys, ts, ps

    def load_data(self, data_path):
        try:
            self.data = np.load(data_path)
            self.xs, self.ys, self.ps, self.ts = self.data[:, 0], self.data[:, 1], self.data[:, 2]*2-1, self.data[:, 3]*1e-6
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))
        print(self.ps)

        if self.sensor_resolution is None:
            self.sensor_resolution = [np.max(self.xs), np.max(self.ys)]
            print("Inferred resolution as {}".format(self.sensor_resolution))
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = False
        self.has_frames = False
        self.t0 = self.ts[0]
        self.tk = self.ts[-1]
        self.num_events = len(self.xs)
        self.num_frames = 0
        self.frame_ts = []

    def find_ts_index(self, timestamp):
        idx = np.searchsorted(self.ts, timestamp)
        return idx

    def ts(self, index):
        return ts[index]

    def compute_frame_indices(self):
        return None

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    args = parser.parse_args()

    dloader = NpyDataset(args.path)
    for item in dloader:
        print(item['events'].shape)
