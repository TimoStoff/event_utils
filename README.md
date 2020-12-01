# event_utils
Event based vision utility library.

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

### `objectives.py`
This file implements various objective functions described in this thesis as well as some other commonly cited works.
Objective functions inherit from the parent class `objective_function}.
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
* `variance_objective`: Variance objective (see \cite{Gallego17ral}).
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
* `xyztheta_warp`: 4-DoF warping function from [Event-based moving object detection and tracking](https://arxiv.org/abs/1803.04523) (x,y,z) velocity and angular velocity theta around the origin).
* `pure_rotation_warp`: 3-DoF pure rotation warp (`x,y,theta` where `x,y` are the center of rotation and `theta` is the angular velocity).

Coming soon. For now, use the code in https://github.com/TimoStoff/events_contrast_maximization for dataset conversion code etc.
