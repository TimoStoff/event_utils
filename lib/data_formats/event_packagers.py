from abc import ABCMeta, abstractmethod
import h5py
import cv2 as cv
import numpy as np

class packager():
    """
    Abstract base class for classes that package event-based data to
    some storage format
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, output_path, max_buffer_size=1000000):
        """
        Set class attributes
        @param name The name of the packager (eg: txt_packager)
        @param output_path Where to dump event data
        @param max_buffer_size For packagers that buffer data prior to
            writing, how large this buffer may maximally be
        """
        self.name = name
        self.output_path = output_path
        self.max_buffer_size = max_buffer_size

    @abstractmethod
    def package_events(self, xs, ys, ts, ps):
        """
        Given events, write them to the file/store them into the buffer
        @param xs x component of events
        @param ys y component of events
        @param ts t component of events
        @param ps p component of events
        @returns None
        """
        pass

    @abstractmethod
    def package_image(self, frame, timestamp):
        """
        Given an image, write it to the file/buffer
        @param frame The image frame to write to the file/buffer
        @param timestamp The timestamp of the frame
        @returns None
        """
        pass

    @abstractmethod
    def package_flow(self, flow, timestamp):
        """
        Given an optic flow image, write it to the file/buffer
        @param frame The optic flow image frame to write to the file/buffer
        @param timestamp The timestamp of the optic flow frame
        @returns None
        """
        pass

    @abstractmethod
    def add_metadata(self, num_events, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow):
        """
        Add metadata to the file
        @param num_events The number of events in the sequence
        @param num_pos The numer of positive events in the sequence
        @param num_neg The numer of negative events in the sequence
        @param duration The length of the sequence in seconds
        @param t0 The start time of the sequence
        @param tk The end time of the sequence
        @param num_imgs The number of images in the sequence
        @param num_flow The number of optic flow frames in the sequence
        """
        pass

    @abstractmethod
    def set_data_available(self, num_images, num_flow):
        """
        Configure the file/buffers depending on which data needs to be written
        @param num_images How many images in the dataset
        @param num_flow How many optic flow frames in the dataset
        """
        pass

class hdf5_packager(packager):
    """
    This class packages data to hdf5 files
    """
    def __init__(self, output_path, max_buffer_size=1000000):
        packager.__init__(self, 'hdf5', output_path, max_buffer_size)
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')
        self.event_xs = self.events_file.create_dataset("events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ys = self.events_file.create_dataset("events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ts = self.events_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.event_ps = self.events_file.create_dataset("events/ps", (0, ), dtype=np.dtype(np.bool_), maxshape=(None, ), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset("images/image{:09d}".format(img_idx),
                data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr" 

    def package_flow(self, flow_image, timestamp, flow_idx):
        flow_dset = self.events_file.create_dataset("flow/flow{:09d}".format(flow_idx),
                data=flow_image, dtype=np.dtype(np.float32))
        flow_dset.attrs['size'] = flow_image.shape
        flow_dset.attrs['timestamp'] = timestamp

    def add_event_indices(self):
        datatypes = ['images', 'flow']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                s = 0
                added = 0
                ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                for image in self.events_file[datatype]:
                    img_ts = self.events_file[datatype][image].attrs['timestamp']
                    event_idx = np.searchsorted(ts, img_ts)
                    if event_idx == len(ts):
                        added += len(ts)
                        s += self.max_buffer_size
                        ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                        event_idx = np.searchsorted(ts, img_ts)
                    event_idx = max(0, event_idx-1)
                    self.events_file[datatype][image].attrs['event_idx'] = event_idx + added

    def add_metadata(self, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow, sensor_size):
        self.events_file.attrs['num_events'] = num_pos+num_neg
        self.events_file.attrs['num_pos'] = num_pos
        self.events_file.attrs['num_neg'] = num_neg
        self.events_file.attrs['duration'] = tk-t0
        self.events_file.attrs['t0'] = t0
        self.events_file.attrs['tk'] = tk
        self.events_file.attrs['num_imgs'] = num_imgs
        self.events_file.attrs['num_flow'] = num_flow
        self.events_file.attrs['sensor_resolution'] = sensor_size
        self.add_event_indices()

    def set_data_available(self, num_images, num_flow):
        if num_images > 0:
            self.image_dset = self.events_file.create_group("images")
            self.image_dset.attrs['num_images'] = num_images
        if num_flow > 0:
            self.flow_dset = self.events_file.create_group("flow")
            self.flow_dset.attrs['num_images'] = num_flow

