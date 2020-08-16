import numpy as np
import os
from .base_dataset import BaseVoxelDataset

class MemMapDataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the MemMap events format used at RPG.
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_frame(self, index):
        frame = self.filehandle['images'][index][:, :, 0]
        return frame

    def get_flow(self, index):
        flow = self.filehandle['optic_flow'][index]
        return flow

    def get_events(self, idx0, idx1):
        xy = self.filehandle["xy"][idx0:idx1]
        xs = xy[:, 0].astype(np.float32)
        ys = xy[:, 1].astype(np.float32)
        ts = self.filehandle["t"][idx0:idx1]
        ps = self.filehandle["p"][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path, timestamp_fname="timestamps.npy", image_fname="images.npy",
                  optic_flow_fname="optic_flow.npy", optic_flow_stamps_fname="optic_flow_stamps.npy",
                  t_fname="t.npy", xy_fname="xy.npy", p_fname="p.npy"):

        assert os.path.isdir(data_path), '%s is not a valid data_path' % data_path

        data = {}
        self.has_flow = False
        for subroot, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                path = os.path.join(subroot, fname)
                if fname.endswith(".npy"):
                    if fname.endswith(timestamp_fname):
                        frame_stamps = np.load(path)
                        data["frame_stamps"] = frame_stamps
                    elif fname.endswith(image_fname):
                        data["images"] = np.load(path, mmap_mode="r")
                    elif fname.endswith(optic_flow_fname):
                        data["optic_flow"] = np.load(path, mmap_mode="r")
                        self.has_flow = True
                    elif fname.endswith(optic_flow_stamps_fname):
                        optic_flow_stamps = np.load(path)
                        data["optic_flow_stamps"] = optic_flow_stamps

                    try:
                        handle = np.load(path, mmap_mode="r")
                    except Exception as err:
                        print("Couldn't load {}:".format(path))
                        raise err
                    if fname.endswith(t_fname):  # timestamps
                        data["t"] = handle.squeeze()
                    elif fname.endswith(xy_fname):  # coordinates
                        data["xy"] = handle.squeeze()
                    elif fname.endswith(p_fname):  # polarity
                        data["p"] = handle.squeeze()
            if len(data) > 0:
                data['path'] = subroot
                if "t" not in data:
                    print("Ignoring root {} since no events".format(subroot))
                    continue
                assert (len(data['p']) == len(data['xy']) and len(data['p']) == len(data['t']))

                self.t0, self.tk = data['t'][0], data['t'][-1]
                self.num_events = len(data['p'])
                self.num_frames = len(data['images'])

                self.frame_ts = []
                for ts in data["frame_stamps"]:
                    self.frame_ts.append(ts)
                data["index"] = self.frame_ts

        self.filehandle = data
        self.find_config(data_path)

    def find_ts_index(self, timestamp):
        index = np.searchsorted(self.filehandle["t"], timestamp)
        return index

    def ts(self, index):
        return self.filehandle["t"][index]

    def infer_resolution(self):
        if len(self.filehandle["images"]) > 0:
            sr = self.filehandle["images"][0].shape[0:2]
        else:
            sr = [np.max(self.filehandle["xy"][:, 1]) + 1, np.max(self.filehandle["xy"][:, 0]) + 1]
            print("Inferred sensor resolution: {}".format(self.sensor_resolution))
        return sr

    def find_config(self, data_path):
        if self.sensor_resolution is None:
            config = os.path.join(data_path, "dataset_config.json")
            if os.path.exists(config):
                self.config = read_json(config)
                self.data_source = self.config['data_source']
                self.sensor_resolution = self.config["sensor_resolution"]
            else:
                data_source = 'unknown'
                self.sensor_resolution = self.infer_resolution()
