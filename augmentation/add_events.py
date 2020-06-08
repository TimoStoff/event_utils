import numpy as np

def add_events(xs, ys, ts, ps, proportion):
    """
    Add events:
        1: Create voxel grid
        2: Norm voxel grid. Voxels are now probabilities
        3: For each event, sample its probability p. p*proportion is now the
            probability of spawning a new event (in a Gaussian around
            the event).
    """
    pass

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to event file")
    parser.add_argument("--output_path", default="/tmp/extracted_data", help="Folder where to put augmented events")
    parser.add_argument("--to_add", type=float, default=1.0, help="Roughly how many more events, as a proportion
        (eg, 1.5 will results in approximately 150% more events.")
    parser.add_argument('--exact', action='store_true', help="If true, will create exactly --to_add events")
    args = parser.parse_args()

