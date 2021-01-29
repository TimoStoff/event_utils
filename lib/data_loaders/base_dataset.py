from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
import random
import os

# local modules
from .data_augmentation import Compose, RobustNorm, CenterCrop
from .data_util import data_sources
from ..representations.voxel_grid import events_to_voxel_torch, events_to_neg_pos_voxel_torch
from ..util.util import read_json, write_json

class BaseVoxelDataset(Dataset):
    """
    Dataloader for voxel grids given file containing events.
    Also loads time-synchronized frames and optic flow if available.
    Voxel grids are formed on-the-fly.
    For each index, returns a dict containing:
        * frame is a H x W tensor containing the first frame whose
          timestamp >= event tensor
        * events is a C x H x W tensor containing the voxel grid
        * flow is a 2 x H x W tensor containing the flow (displacement) from
          the current frame to the last frame
        * dt is the time spanned by 'events'
        * data_source_idx is the index of the data source (simulated, IJRR, MVSEC etc)
    Subclasses must implement:
        - get_frame(index) method which retrieves the frame at index i
        - get_flow(index) method which retrieves the optic flow at index i
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            has_flow - if this dataset has optic flow
            t0 - timestamp of first event
            tk - timestamp of last event
            num_events - the total number of events
            frame_ts - list of the timestamps of the frames
            num_frames - the number of frames
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path Path to the file containing the event/image data
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            * "fixed_frames" ('num_frames' voxels formed at even intervals)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_seconds', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            method={'method':'fixed_frames', 'num_frames':100}
            Default is 'between_frames'.
    """

    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        raise NotImplementedError

    def get_flow(self, index):
        """
        Get optic flow at index
        @param index The index of the optic flow to get
        """
        raise NotImplementedError

    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        @param idx0 Start index to get events from
        @param idx1 End index to get events from
        """
        raise NotImplementedError

    def load_data(self, data_path):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.has_flow - if this dataset has optic flow
            self.t0 - timestamp of first event
            self.tk - timestamp of last event
            self.num_events - the total number of events
            self.frame_ts - list of the timestamps of the frames
            self.num_frames - the number of frames
        @param data_path The path to the data file/s containing events etc
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        @param timestamp The timestamp at which to find the corresponding event index
        """
        raise NotImplementedError

    def ts(self, index):
        """
        Get timestamp at index
        @param Index of event whose timestamp to return
        """
        raise NotImplementedError

    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method={'method': 'between_frames'}, max_length=None, combined_voxel_channels=False,
                 return_events=False, return_voxelgrid=True, return_frame=True, return_prev_frame=False,
                 return_flow=True, return_prev_flow=False, return_format='torch'):
        """
        @param data_path Path to the file containing the event/image data
        @param transforms Dict containing the desired augmentations
        @param sensor_resolution The size of the image sensor from which the events originate
        @param num_bins The number of bins desired in the voxel grid
        @param voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events, with each batch
                overlapping by 'sliding_window_w' events)
            * "t_seconds" (new voxels are formed every t seconds, with each batch
                overlapping by 'sliding_window_t' seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            * "fixed_frames" ('num_frames' voxels formed at even intervals)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_seconds', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            method={'method':'fixed_frames', 'num_frames':100}
            Default is 'between_frames'.
        @param max_length Maximum capped length of dataset (no cap if left empty)
        @param combined_voxel_channels If True, produces one voxel grid for all events, if False,
            produces separate voxel grids for positive and negative channels
        @param return_events If true, returns events in output dict
        @param return_voxelgrid If true, returns voxelgrid in output dict
        @param return_frame If true, returns frames in output dict
        @param return_prev_frame If true, returns previous frame to current frame
            in output dict
        @param return_flow If true, returns optic flow in output dict
        @param return_prev_flow If true, returns previous optic flow to current
            optic flow in output dict
        @param return_format The desired output format (options = 'numpy' and 'torch')
        """

        self.num_bins = num_bins
        self.data_path = data_path
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.data_source_idx = -1
        self.has_flow = False
        self.has_frames = True
        self.return_format = return_format
        self.counter = 0

        self.return_events = return_events
        self.return_voxelgrid = return_voxelgrid
        self.return_frame = return_frame
        self.return_prev_frame = return_prev_frame
        self.return_flow = return_flow
        self.return_prev_flow = return_prev_flow

        self.sensor_resolution, self.t0, self.tk, self.num_events, self.frame_ts, self.num_frames = \
            None, None, None, None, None, None

        self.load_data(data_path)

        if self.sensor_resolution is None or self.has_flow is None or self.t0 is None \
                or self.tk is None or self.num_events is None or self.frame_ts is None \
                or self.num_frames is None:
            print("s_r: {}, h_f={}, t0={}, tk={}, n_e={}, nf={}, s_f={}".format(self.sensor_resolution is None, self.has_flow is None, self.t0 is None, self.tk is None, self.num_events is None, self.frame_ts is None, self.num_frames))
            raise Exception("Dataloader failed to intialize all required members")

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        self.duration = self.tk - self.t0

        self.set_voxel_method(voxel_method)

        self.normalize_voxels = False
        if 'RobustNorm' in transforms.keys():
            vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
            del (transforms['RobustNorm'])
            self.normalize_voxels = True
            self.vox_transform = Compose(vox_transforms_list)

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

    @staticmethod
    def preprocess_events(xs, ys, ts, ps):
        """
        Given empty events, return single zero event
        @param xs x compnent of events
        @param ys y compnent of events
        @param ts t compnent of events
        @param ps p compnent of events
        """
        if len(xs) == 0:
            txs = np.zeros((1))
            tys = np.zeros((1))
            tts = np.zeros((1))
            tps = np.zeros((1))
            return txs, tys, tts, tps
        return xs, ys, ts, ps

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
        @param index Index of data
        @param seed Random seed for data augmentation
        @returns Dict with desired outputs (voxel grid, events, frames etc)
            as set in constructor
        """
        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed

        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        xs, ys, ts, ps = self.preprocess_events(xs, ys, ts, ps)
        ts_0, ts_k  = ts[0], ts[-1]
        dt = ts_k-ts_0

        item = {'data_source_idx': self.data_source_idx, 'data_path': self.data_path,
                'timestamp': ts_k, 'dt_between_frames': dt, 'ts_idx0': ts_0, 'ts_idx1': ts_k,
                'idx0': idx0, 'idx1': idx1}
        if self.return_voxelgrid:
            voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
            voxel = self.transform_voxel(voxel, seed)
            item['voxel'] = voxel

        if self.voxel_method['method'] == 'between_frames':
            frame = self.get_frame(index)
            frame = self.transform_frame(frame, seed)

            if self.has_flow:
                flow = self.get_flow(index)
                # convert to displacement (pix)
                flow = flow * dt
                flow = self.transform_flow(flow, seed)
            else:
                if self.return_format == 'torch':
                    flow = torch.zeros((2, frame.shape[-2], frame.shape[-1]), dtype=frame.dtype, device=frame.device)
                else:
                    flow = np.zeros((2, frame.shape[-2], frame.shape[-1]))

            if self.return_flow:
                item['flow'] = flow
                item['flow_ts'] = self.frame_ts[index]
            if self.return_prev_flow:
                prev_flow = flow if not self.has_flow else self.get_flow(index)
                item['prev_flow'] = self.transform_flow(prev_flow, seed)
            if self.return_frame:
                item['frame'] = frame
                item['frame_ts'] = self.frame_ts[index]
            if self.return_prev_frame:
                item['prev_frame'] = self.transform_frame(self.get_frame(index), seed)
        else:
            frames = []
            frame_ts = []
            if self.has_frames and self.return_frame:
                fi = self.frame_indices[index]
                if fi[0] != -1:
                    frames = [self.transform_frame(self.get_frame(fidx), seed) for fidx in range(fi[1]-fi[0])]
                    frame_ts = self.frame_ts[fi[0]:fi[1]]
            item['frame'] = frames
            item['frame_ts'] = frame_ts

            flows = []
            flow_ts = []
            if self.has_flow and self.return_flow:
                fi = self.frame_indices[index]
                if fi[0] != -1 and self.has_flow:
                    flows = [self.transform_flow(self.get_flow(fidx), seed) for fidx in range(fi[0], fi[1], 1)]
                    flow_ts = self.frame_ts[fi[0]:fi[1]]
            item['flow'] = flows
            item['flow_ts'] = flow_ts

        if self.return_events:
            if self.return_format == 'torch':
                if idx0-idx1 == 0:
                    item['events'] = torch.zeros((1, 4), dtype=torch.float32)
                    item['events_batch_indices'] = torch.ones((1))
                    item['ts_idx0'] = torch.zeros((1), dtype=torch.float64)
                else:
                    item['events'] = torch.from_numpy(np.stack((xs, ys, ts-ts_0, ps), axis=1)).float()
                    item['events_batch_indices'] = idx1-idx0
                    item['ts_idx0'] = torch.tensor(ts_0)
            elif self.return_format == 'numpy':
                if idx0-idx1 == 0:
                    item['events'] = np.zeros((1, 4))
                    item['events_batch_indices'] = np.ones((1))
                    item['ts_idx0'] = np.zeros((1))
                else:
                    item['events'] = np.stack((xs, ys, ts, ps), axis=1)
                    item['events_batch_indices'] = idx1-idx0
                    item['ts_idx0'] = np.array(ts_0)
            else:
                raise Exception("Invalid event format '{}' used".format(self.return_format))
        return item

    def compute_between_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        @returns List of indices of events at each frame timestamp
        """
        frame_indices = []
        start_idx = 0
        for ts in self.frame_ts:
            end_index = self.find_ts_index(ts)
            if end_index >= self.num_events:
                end_index = self.num_events-1
            frame_indices.append([start_idx, end_index])
            start_idx = end_index
        return frame_indices

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_seconds), find the start and
        end indices of the corresponding events
        @returns List of indices of events at beginning and end of each block of time
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_k_indices(self):
        """
        For each block of k events, find the start and
        end indices of the corresponding events
        @returns List of indices of events at beginning and end of each block of
            k events (with sliding window)
        """
        k_indices = []
        start_idx = 0
        for i in range(self.__len__()):
            idx0 = (self.voxel_method['k'] - self.voxel_method['sliding_window_w']) * i
            idx1 = idx0 + self.voxel_method['k']
            k_indices.append([idx0, idx1])
        return k_indices

    def compute_per_frame_indices(self):
        """
        For each set of event_indices, find the enclosed frame indices
        @returns List of frame indices at each event index
        """
        frame_indices = []
        for indices in self.event_indices:
            s_t, e_t = self.ts(int(indices[0])), self.ts(int(indices[1]))
            idx0 = min(np.searchsorted(self.frame_ts, s_t), len(self.frame_ts)-1)
            idx1 = min(np.searchsorted(self.frame_ts, e_t), len(self.frame_ts)-1)
            if idx0 == idx1:
                frame_indices.append([-1, -1])
            else:
                frame_indices.append([idx0, idx1])
        return frame_indices

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        @param voxel_method The method of voxel formation as set in constructor.
            Options = {'k_events', 't_seconds, 'fixed_frames', 'between_frames'}
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'k_events':
            self.length = max(int(self.num_events / (voxel_method['k'] - voxel_method['sliding_window_w'])), 0)
            if self.length == 0:
                print("num_events={}, t={}, window={}".format(self.num_events, voxel_method['k'], voxel_method['sliding_window_w']))
            self.event_indices = self.compute_k_indices()
        elif self.voxel_method['method'] == 't_seconds':
            self.length = max(int(self.duration / (voxel_method['t'] - voxel_method['sliding_window_t'])), 0)
            if self.length == 0:
                print("duration={}, t={}, window={}".format(self.duration, voxel_method['t'], voxel_method['sliding_window_t']))
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'fixed_frames':
            self.length = self.voxel_method['num_frames']
            self.voxel_method['t'] = (self.tk-self.t0)/self.length
            voxel_method['sliding_window_t'] = 0
            self.event_indices = self.compute_timeblock_indices()
        elif self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - 1
            self.event_indices = self.compute_between_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        print("Dataset contains {} items".format(self.length))
        if self.has_frames:
            self.frame_indices = self.compute_per_frame_indices()
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        @param Desired data index
        @returns Start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return int(idx0), int(idx1)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        @param xs tensor containg x coords of events
        @param ys tensor containg y coords of events
        @param ts tensor containg t coords of events
        @param ps tensor containg p coords of events
        @param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        @returns Voxel grid of input events
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        @param frame Input frame
        @param seed  Seed for random number generation
        @returns Augmented frame
        """
        if self.return_format == "torch":
            frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        @param flow Input flow
        @param seed  Seed for random number generation
        @returns Augmented flow
        """
        if self.return_format == "torch":
            flow = torch.from_numpy(flow)  # should end up [2 x H x W]
            if self.transform:
                random.seed(seed)
                flow = self.transform(flow, is_flow=True)
        return flow

    def size(self):
        """
        Get the size of the event camera sensor/resolution
        @returns Sensor resolution
        """
        return self.sensor_resolution

    @staticmethod
    def unpackage_events(events):
        """
        Given events as 2D array, break it up into xs,ys,ts,ps components
        @returns xs, ys, ts, ps component of events
        """
        return events[:,0], events[:,1], events[:,2], events[:,3]

    @staticmethod
    def collate_fn(data, event_keys=['events'], idx_keys=['events_batch_indices']):
        """
        Custom collate function for pyTorch batching to allow batching events
        """
        collated_events = {}
        events_arr = []
        end_idx = 0
        batch_end_indices = []
        for idx, item in enumerate(data):
            for k, v in item.items():
                if not k in collated_events.keys():
                    collated_events[k] = []
                if k in event_keys:
                    end_idx += v.shape[0]
                    events_arr.append(v)
                    batch_end_indices.append(end_idx)
                else:
                    collated_events[k].append(v)
        for k in collated_events.keys():
            try:
                i = event_keys.index(k)
                events = torch.cat(events_arr, dim=0)
                collated_events[event_keys[i]] = events
                collated_events[idx_keys[i]] = batch_end_indices
            except:
                collated_events[k] = default_collate(collated_events[k])
        return collated_events
