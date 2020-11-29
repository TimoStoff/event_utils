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
* `add_random_events`: Generates $N$ new random events, drawn from a uniform distribution over the size of the spatiotemporal volume.
* `remove_events`: Makes the event stream more sparse, by removing a random selection of $N$ events from the original event stream.
* `add_correlated_events`: Makes the event stream more dense by adding $N$ new events around the existing events.
Each original event is fitted with a Gaussian bubble with standard deviation $\sigma_{xy}$ in the $x,y$ dimension and $\sigma_{t}$ in the $t$ dimension.
New events are drawn from these distributions.
Note that this also 'blurs' the event stream.
* `flip_events_x`: Flip events over x axis.
* `flip_events_y`: Flip events over y axis.
* `crop_events`: Spatially crop events either randomly, to a desired amount and either from the origin or as a center crop.
* `rotate_events`: Rotate events by angle $\theta$ around a center of rotation $a,b$.
Events can then optionally be cropped in the case that they overflow the sensor resolution.
Some possible augmentations are shown below:
Since the augmentations are implemented using vectorisation, the heavy lifting is done in optimised C/C++ backends and is thus very fast.


Coming soon. For now, use the code in https://github.com/TimoStoff/events_contrast_maximization for dataset conversion code etc.
