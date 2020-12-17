# event_utils
Event based vision utility library. For additional detail, see the thesis document [Motion Estimation by Focus Optimisation: Optic Flow and Motion Segmentation with Event Cameras](https://timostoff.github.io/thesis). If you use this code in an academic context, please cite:
```
@PhDThesis{Stoffregen20Thesis,
  author        = {Timo Stoffregen},
  title         = {Motion Estimation by Focus Optimisation: Optic Flow and Motion Segmentation with Event Cameras},
  school        = {Department of Electrical and Computer Systems Engineering, Monash University},
  year          = 2020
}
```

This is an event based vision utility library with functionality for focus optimisation, deep learning, event-stream noise augmentation, data format conversion and efficient generation of various event representations (event images, voxel grids etc).

The library is implemented in Python. Nevertheless, the library is efficient and fast, since almost all of the hard work is done using vectorisation or numpy/pytorch functions. All functionality is implemented in numpy _and_ pytorch, so that on-GPU processing for hardware accelerated performance is very easy. 

The library is divided into eight sub-libraries:
```
└── lib
    ├── augmentation
    ├── contrast_max
    ├── data_formats
    ├── data_loaders
    ├── representations
    ├── transforms
    ├── util
    └── visualization
```

## augmentation
While the `data_loaders` learning library contains some code for tensor augmentation (such as adding Gaussian noise, rotations, flips, random crops etc), the augmentation library allows for these operations to occur on the raw events.
This functionality is contained within `event_augmentation.py`.
### `event_augmentation.py`
The following augmentations are available:
* `add_random_events`: Generates N new random events, drawn from a uniform distribution over the size of the spatiotemporal volume.
* `remove_events`: Makes the event stream more sparse, by removing a random selection of N events from the original event stream.
* `add_correlated_events`: Makes the event stream more dense by adding N new events around the existing events.
Each original event is fitted with a Gaussian bubble with standard deviation `sigma_xy` in the `x,y` dimension and `sigma_t` in the `t` dimension.
New events are drawn from these distributions.
Note that this also 'blurs' the event stream.
* `flip_events_x`: Flip events over x axis.
* `flip_events_y`: Flip events over y axis.
* `crop_events`: Spatially crop events either randomly, to a desired amount and either from the origin or as a center crop.
* `rotate_events`: Rotate events by angle `theta` around a center of rotation `a,b`.
Events can then optionally be cropped in the case that they overflow the sensor resolution.
Some possible augmentations are shown below:
Since the augmentations are implemented using vectorisation, the heavy lifting is done in optimised C/C++ backends and is thus very fast.

![Augmentation examples](https://github.com/TimoStoff/event_utils/blob/master/.images/augmentation.png)

Some examples of augmentations on the `slider_depth` sequence from the [event camera dataset](http://rpg.ifi.uzh.ch/davis_data.html) can be seen above (events in red and blue with the first events in black to show scene structure). (a) the original event stream, (b) doubling the events by adding random _correlated_ events, (c) doubling the events by adding fully random (normal distribution) events, (d) halving the events by removing random, (e) flipping the events horizontally, (f) rotating the events 45 degrees. Demo code to reproduce these plots can be found by executing the following (note that the events need to be in HDF5 format):
```python lib/augmentation/event_augmentation.py /path/to/slider_depth.h5 --output_path /tmp```

## contrast_max
The focus optimisation library contains code that allows the user to perform focus optimisation on events.
The important files of this library are:
`events_cmax.py`
This file contains code to perform focus optimisation.
The most important functionality is provided by:
* `grid_search_optimisation`: Performs the grid search optimisation from [SOFAS algorithm](https://arxiv.org/abs/1805.12326).
* `optimize`: Performs gradient based focus optimisation on the input events, given an objective function and motion model.
* `grid_cmax`: Given a set of events, splits the image plane into ROI of size `roi_size`.
	Performs focus optimisation on each ROI separately.
* `segmentation_mask_from_d_iwe`: Retrieve a segmentation mask for the events based on dIWE/dWarpParams.
* `draw_objective_function`: Draw the objective function for a given set of events, motion model and objective function.
Produces plots as in below image.
* `main`: Demo showing various capabilities and code examples.

![Focus Optimisation](https://github.com/TimoStoff/event_utils/blob/master/.images/cmax.png)

Examples can be seen in the images above: each set of events is drawn with the variance objective function (w.r.t. optic flow motion model) underneath. This set of tools allows optimising the objective function to recover the motion parameters (images generated with the library). 

### `objectives.py`
This file implements various objective functions described in this thesis as well as some other commonly cited works.
Objective functions inherit from the parent class `objective_function`.
The idea is to make it as easy as possible to add new, custom objective functions by providing a common API for the optimisation code.
This class has several members that require initialisation:
* `name`: The name of the objective function (eg `variance`).
* `use_polarity`: Whether to use the polarity of the events in generating IWEs.
* `has_derivative`: Whether this objective has an analytical derivative w.r.t. warp parameters.
* `default_blur`: What `sigma` should be default for blurring.
* `adaptive_lifespan`: An innovative feature to deal with linearisation errors. 
	Many implementations of contrast maximisation use assumptions of linear motion w.r.t. the chosen motion model. 
	A given estimate of the motion parameters implies a lifespan of the events. 
	If `adaptive_lifespan`: is True, the number of events used during warping is cut to that lifespan for each optimisation step, computed using `pixel_crossings`.
	eg If motion model is optic flow velocity and the estimate = 12 pixels/second and `pixel_crossings`=3, then the lifespan will be 3/12=0.25s.
* `pixel_crossings`: Number of pixel crossings used to calculate lifespan.
* `minimum_events`: The minimal number of events that the lifespan can cut to.
The required function that inheriting classes need to implement are:
* `evaluate_function`: Evaluate the objective function for given parameters, events etc.
* `evaluate_gradient`: Evaluate the objective function and the gradient of the objective function w.r.t. motion parameters for given parameters, events etc.
The objective functions implemented in this file are:
* `variance_objective`: Variance objective (see [Accurate Angular Velocity Estimation with an Event Camera](https://www.zora.uzh.ch/id/eprint/138896/1/RAL16_Gallego.pdf)).
* `rms_objective`: Root Mean Squared objective.
* `sos_objective`: See [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `soe_objective`: See [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `moa_objective`: See [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `soa_objective`: See [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `sosa_objective`: See [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `zhu_timestamp_objective`: Objective function defined in [Unsupervised event-based learning of optical flow, depth, and egomotion](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.pdf).
* `r1_objective`: Combined objective function R1 [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)
* `r2_objective`: Combined objective function R2 [Event Cameras, Contrast Maximization and Reward Functions: An Analysis](https://openaccess.thecvf.com/content_CVPR_2019/html/Stoffregen_Event_Cameras_Contrast_Maximization_and_Reward_Functions_An_Analysis_CVPR_2019_paper.html)

### `warps.py`
This file implements warping functions described in this thesis as well as some other commonly cited works.
Objective functions inherit from the parent class `warp_function`.
The idea is to make it as easy as possible to add new, custom warping functions by providing a common API for the optimisation code.
Initialisation requires setting member variables:
* `name`: Name of the warping function, eg `optic_flow`.
* `dims`: DoF of the warping function.
The only function that needs to be implemented by inheriting classes is `warp`, which takes events, a reference time and motion parameters as input.
The function then returns a list of the warped event coordinates as well as the Jacobian of each event w.r.t. the motion parameters.
Warp functions currently implemented are:
* `linvel_warp`: 2-DoF optic flow warp.
* `xyztheta_warp`: 4-DoF warping function from [Event-based moving object detection and tracking](https://arxiv.org/abs/1803.04523) (`x,y,z`) velocity and angular velocity `theta` around the origin).
* `pure_rotation_warp`: 3-DoF pure rotation warp (`x,y,theta` where `x,y` are the center of rotation and `theta` is the angular velocity).

## `data_formats`
The `data_formats` provides code for converting events in one file format to another.
Even though many candidates have appeared over the years (rosbag, AEDAT, .txt, `hdf5`, pickle, cuneiform clay tablets, just to name a few), a universal storage option for event based data has not yet crystallised.
Some of these data formats are particularly useful within particular operating systems or programming languages.
For example, rosbags are the natural choice for C++ programming with the `ros` environment.
Since they also store data in an efficient binary format, they have become a very common storage option.
However, they are notoriously slow and impractical to process in Python, which has become the de-facto deep-learning language and is commonly used in research due to the rapid development cycle.
More practical (and importantly, fast) options are the `hdf5` and numpy memmap formats.
`hdf5` is a more compact and easily accessible format, since it allows for easy grouping and metadata allocation, however it's difficulty in setting up multi-threading access and subsequent buggy behaviour (even in read-only applications) means that memmap is more common for deep learning, where multi-threaded data-loaders can significantly speed up training.

### `event_packagers.py`
The `data_formats` library provides a `packager` abstract base class, which defines what a `packager` needs to do.
`packager`objects receive data (events, frames etc) and write them to the desired file format (eg `hdf5`).
Converting file formats is now much easier, since input files now need only to be parsed and the data sent to the `packager`with the appropriate function calls.
The functions that need to implemented are:
* `package_events` A function which given events, writes them to the file/buffer.
* `package_image` A function which given images, writes them to the file/buffer.
* `package_flow` A function which given optic flow frames, writes them to the file/buffer.
* `add_metadata` Writes metadata to the file (number of events, number of negative/positive events, duration of sequence, start time, end time, number of images, number of optic flow frames).
* `set_data_available` What data is available and needs to be written (ie events, frames, optic flow).
A `packager` for `hdf5` and memmap is implemented.
### `h5_to_memmap.py` and `rosbag_to_h5.py`
The library implements two converters, one for `hdf5` to memmap and one for rosbag to `hdf5`.
These can be easily called from the command line with various options that can be found in the documentation.
### `add_hdf5_attribute.py`
`add_hdf5_attribute.py` allows the user to add or modify attributes to existing `hdf5` files.
Attributes are the manner in which metadata is saved in `hdf5` files.
### `read_events.py`
`read_events.py` contains functions for reading events from `hdf5` and memmap.
The functions are:
* `read_memmap_events`.
* `read_h5_events`.

## `data_loader`
The deep learning code can be found in the `data_loaders`library.
It contains code for loading events and transforming them into voxel grids in an efficient manner as well as code for data augmentation.
Actual networks and cost functions described in this thesis are not implemented in the library but at the project page for that paper.

`data_loaders` provides a highly versatile `pytorch` dataloader, which can be used across various storage formats for events (.txt, `hdf5`, memmap etc).
As a result it is very easy to implement new dataloader for a different storage format.
The output of the dataloader was originally to provide voxel grids of the events, but can be used just as well to output batched events, due to a custom `pytorch`collation function.
As a result, the dataloader is useful for any situation in which it is desirable to iterate over the events in a storage medium and is not only useful for deep learning.
For instance, if one wants to iterate over the events that lie between all the frames of a `davis` sequence, the following code is sufficient:
```
dloader = DynamicH5Dataset(path_to_events_file)
for item in dloader:
	print(item[`events'].shape)
```

### `base_dataset.py`
This file defines the base dataset class (`BaseVoxelDataset`), which defines all batching, augmentation, collation and housekeeping code.
Inheriting classes (one per data format) need only to implement the abstract functions for providing events, frames and other data from storage.
These abstract functions are:
* `get_frame(self, index)` Given an index `n`, return the `n`th frame.
* `get_flow(self, index)` Given an index `n`, return the `n`th optic flow frame.
* `get_events(self, idx0, idx1)` Given a start and end index `idx0` and `idx1`, return all events between those indices.
* `load_data(self, data_path)` Function which is called once during initialisation, which creates handles to files and sets several class attributes (number of frames, events etc).
* `find_ts_index(self, timestamp)` Given a timestamp, get the index of the nearest event.
* `ts(self, index)` Given an event index, return the timestamp of that event.
The function `load_data`must set the following member variables:
* `self.sensor_resolution` Event sensor resolution.
* `self.has_flow` Whether or not the data has optic flow frames.
* `self.t0` The start timestamp of the events.
* `self.tk` The end timestamp of the events.
* `self.num_events` The number of events in the dataset.
* `self.frame_ts` The timestamps of the time-synchronised frames.
* `self.num_frames` The number of frames in the dataset.
The constructor of the class takes following arguments:
* `data_path` Path to the file containing the event/image data.
* `transforms` Python dict containing the desired augmentations.
* `sensor_resolution` The size of the image sensor.
* `num_bins` The number of bins desired in the voxel grid.
* `voxel_method` Which method should be used to form the voxels.
* `max_length` If desired, the length of the dataset can be capped to `max_length` batches.
* `combined_voxel_channels` If True, produces one voxel grid for all events, if False, produces separate voxel grids for positive and negative channels.
* `return_events` If true, returns events in output dict.
* `return_voxelgrid` If true, returns voxel grid in output dict.
* `return_frame` If true, returns frames in output dict.
* `return_prev_frame` If true, returns previous batch's frame to current frame in output dict.
* `return_flow` If true, returns optic flow in output dict.
* `return_prev_flow` If true, returns previous batch's optic flow to current optic flow in output dict.
* `return_format` Which output format to use (options=`'numpy'` and `'torch'`).
The parameter `voxel_method` defines how the data is to be batched.
For instance, one might wish to have data returned in windows `t` seconds wide, or to always get all data between successive `aps` frames.
The method is given as a dict, as some methods have additional parametrisations.
The current options are:
* `k_events` Data is returned every `k` events.
	The dict is given in the format `method = {'method': 'k_events', 'k': value_for_k, 'sliding_window_w': value_for_sliding_window}`.
	The parameter `sliding_window_w` defines by how many events each batch overlaps.
* `t_seconds` Data is returned every `t` seconds.
	The dict is given in the format `method = {'method': 't_seconds', 't': value_for_t, 'sliding_window_t': value_for_sliding_window}`.
	The parameter `sliding_window_t` defines by how many seconds each batch overlaps.
* `between_frames` All data between successive frames is returned.
	Requires time-synchronised frames to exist.
	The dict is given in the format `method={'method':'between_frames'}`.
Generating the voxel grids can be done very efficiently and on the `gpu` (if the events have been loaded there) using the `pytorch` function `target.index_put_(index, value, accumulate=True)`.
This function puts values from `value` into `target` using the indices specified in `indices` using highly optimised C++ code in the background.
`accumulate` specifies if values in `value` which get put in the same location on `target` should sum (accumulate) or overwrite one another.
In summary, `BaseVoxelDataset` allows for very fast, on-device data-loading and on-the-fly voxel grid generation.

## `representations`
This library contains code for generating representations from the events in a highly efficient, `gpu` ready manner.
![Representations](https://github.com/TimoStoff/event_utils/blob/master/.images/representations.png)
Various representations can be seen above with (a) the raw events, (b) the voxel grid, (c) the event image, (d) the timestamp image.
### `voxel_grid.py`
This file contains several means for forming and viewing voxel grids from events.
There are two versions of each function, representing a pure `numpy` and a `pytorch` implementation.
The `pytorch` implementation is necessary for `gpu` processing, however it is not as commonly used as `numpy`, which is so frequently used as to barely be a dependency any more.
Functions for `pytorch` are:
* `voxel_grids_fixed_n_torch` Given a set of `n` events, return a voxel grid with `B` bins and with a fixed number of events.
* `voxel_grids_fixed_t_torch` Given a set of events and a duration `t`, return a voxel grid with `B` bins and with a fixed temporal width `t`.
* `events_to_voxel_timesync_torch` Given a set of events and two times `t_0` and `t_1`, return a voxel grid with `B` bins from the events between `t_0` and `t_1`.
* `events_to_voxel_torch` Given a set of events, return a voxel grid with `B` bins from those events.
* `events_to_neg_pos_voxel_torch` Given a set of events, return a voxel grid with `B` bins from those events.
Positive and negative events are formed into two separate voxel grids.
Functions for `numpy` are:
* `events_to_voxel` Given a set of events, return a voxel grid with `B` bins from those events.
* `events_to_neg_pos_voxel` Given a set of events, return a voxel grid with `B` bins from those events.
Positive and negative events are formed into two separate voxel grids.
Additionally:
* `get_voxel_grid_as_image`Returns a voxel grid as a series of images, one for each bin for display.
* `plot_voxel_grid` Given a voxel grid, display it as an image.
Voxel grids can be formed both using spatial and temporal interpolation between the bins.
### `image.py`
`image.py` contains code for forming images from events in an efficient manner.
The functions allow for forming images with both discrete and floating point events using bilinear interpolation.
Images currently supported are event images and timestamp images using either `numpy` or `pytorch`.
Functions are:
* `events_to_image` Form an image from events using `numpy`.
Allows for bilinear interpolation while assigning events to pixels and padding of the image or clipping of events for events which fall outside of the range.
* `events_to_image_torch` Form an image from events using `pytorch`.
Allows for bilinear interpolation while assigning events to pixels and padding of the image or clipping of events for events which fall outside of the range.
* `image_to_event_weights` Given an image and a set of event coordinates, get the pixel value of the image for each event using reverse bilinear interpolation.
* `events_to_image_drv` Form an image from events and the derivative images from the event Jacobians (with options for padding the image or clipping out-of-range events).
Of particular use for `cmax` where analytic gradients motion models are known.
* `events_to_timestamp_image` Method to generate the average timestamp images from [Unsupervised event-based learning of optical flow, depth, and egomotion](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.pdf) using `numpy`.
Returns two images, one for negative and one for positive events.
* `events_to_timestamp_image_torch` Method to generate the average timestamp images from [Unsupervised event-based learning of optical flow, depth, and egomotion](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.pdf) using `pytorch`.
Returns two images, one for negative and one for positive events.

## `util`
This library contains some utility functions used in the rest of the library.
Functions include:
* `infer_resolution` Given events, guess the resolution by looking at the max and min values.
* `events_bounds_mask` Get a mask of the events that are within given bounds.
* `clip_events_to_bounds` Clip events to the given bounds.
* `cut_events_to_lifespan` Given motion model parameters, compute the speed and thus the lifespan, given a desired number of pixel crossings.
* `get_events_from_mask` Given an image mask, return the indices of all events at each location in the mask.
* `binary_search_h5_dset` Binary search for a timestamp in an `hdf5` event file, without loading the entire file into RAM.
* `binary_search_torch_tensor` Binary search implemented for `pytorch` tensors (no native implementation exists).
* `remove_hot_pixels` Given a set of events, removes the `hot' pixel events. Accumulates all of the events into an event image and removes the `num_hot` highest value pixels.
* `optimal_crop_size` Find the optimal crop size for a given `max_size` and `subsample_factor`. The optimal crop size is the smallest integer which is greater or equal than `max_size`, while being divisible by 2^`max_subsample_factor`.
* `plot_image_grid` Given a list of images, stitch them into a grid and display/save the grid.
* `flow2bgr_np` Turn optic flow into an RGB image.

## `visualisation`
The `visualization` library contains methods for generating figures and movies from events.
The majority of figures shown in the thesis were generated using this library.
Two rendering backends are available, the commonly used `matplotlib` plotting library and `mayavi`, which is a VTK based graphics library.
The API for both of these is essentially the same, the main difference being the dependency on `matplotlib` or `mayavi`.
`matplotlib` is very easy to set up, but quite slow, `mayavi` is very fast but more difficult to set up and debug.
I will describe the `matplotlib` version here, although all functionality exists in the `mayavi` version too (see the code documentation for details).
### `draw_event_stream.py`
The core work is done in this file, which contains code for visualising events and voxel grids for examples).
The function for plotting events is `plot_events`.
\input{figures/appendix/visualisations/fig.tex}
Input parameters for this function are:
* `xs` x coords of events.
* `ys` y coords of events.
* `ts` t coords of events.
* `ps` p coords of events.
* `save_path` If set, will save the plot to here
* `num_compress` Takes `num_compress` events from the beginning of the sequence and draws them in the plot at time `t=0` in black.
	This aids visibility (see the augmentation examples).
* `compress_front` If True, display the compressed events in black at the front of the spatiotemporal volume rather than the back
* `num_show` Sets the number of events to plot.
	If set to -1 will plot all of the events (can be potentially expensive).
	Otherwise, skips events in order to achieve the desired number of events
* `event_size` Sets the size of the plotted events.
* `elev` Sets the elevation of the plot.
* `azim` Sets the azimuth of the plot.
* `imgs` A list of images to draw into the spatiotemporal volume.
* `img_ts` A list of the position on the temporal axis where each image from `imgs` is to be placed.
* `show_events` If False, will not plot the events (only images).
* `show_plot` If True, display the plot in a `matplotlib` window as well as saving to disk.
* `crop` A crop, if desired, of the events and images to be plotted.
* `marker` Which marker should be used to display the events (default is '.', which results in points, but circles 'o' or crosses 'x' are among many other possible options).
* `stride` Determines the pixel stride of the image rendering (1=full resolution, but can be quite resource intensive).
* `invert` Inverts the colour scheme for black backgrounds.
* `img_size` The size of the sensor resolution. Inferred if empty.
* `show_axes` If True, draw axes onto the plot.
The analogous function for plotting voxel grids is:
* `xs` x coords of events.
* `ys`y coords of events.
* `ts` t coords of events.
* `ps` p coords of events.
* `bins` The number of bins to have in the voxel grid.
* `frames` A list of images to draw into the plot with the voxel grid.
* `frame_ts` A list of the position on the temporal axis where each image from `frames` is to be placed.
* `sensor_size` Event sensor resolution.
* `crop` A crop, if desired, of the events and images to be plotted.
* `elev` Sets the elevation of the plot.
* `azim` Sets the azimuth of the plot.
To plot successive frames in order to generate video, the function `plot_events_sliding` can be used.
Essentially, this function renders a sliding window of the events, for either the event or voxel visualisation modes.
Similarly, `plot_between_frames` can be used to render all events between frames, with the option to skip every `n`th event.
To generate such plots from the command line, the library provides the scripts:
* `visualize_events.py`
* `visualize_voxel.py`
* `visualize_flow.py`
These provide a range of documented commandline arguments with sensble defaults from which plots of the events, voxel grids and events with optic flow overlaid can be generated.
For example,
```python visualize_events.py /path/to/slider_depth.h5```
produces plots of the `slider_depth` sequence.
Invoking:
```python visualize_voxel.py /path/to/slider_depth.h5```
produces voxels of the `slider_depth` sequence.
\input{figures/appendix/slider_vis/fig.tex}
![Visualisation](https://github.com/TimoStoff/event_utils/blob/master/.images/visualisations.png)
Typical visualisations are shown above: the `slider_depth` sequence is drawn as successive frames of events (top) and voxels (bottom).
