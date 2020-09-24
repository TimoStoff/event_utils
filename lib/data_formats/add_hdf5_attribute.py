import argparse
import numpy as np
import h5py
import os
import glob

def endswith(path, extensions):
    for ext in extensions:
        if path.endswith(ext):
            return True
    return False

def get_filepaths_from_path_or_file(path, extensions=[], datafile_extensions=[".txt", ".csv"]):
    files = []
    path = path.rstrip("/")
    if os.path.isdir(path):
        for ext in extensions:
            files += sorted(glob.glob("{}/*{}".format(path, ext)))
    else:
        if endswith(path, extensions):
            files.append(path)
        elif endswith(path, datafile_extensions):
            with open(path, 'r') as f:
                #files.append(line) for line in f.readlines
                files = [line.strip() for line in f.readlines()]
    return files

def add_attribute(h5_filepaths, group, attribute_name, attribute_value, dry_run=False):
    for h5_filepath in h5_filepaths:
        print("adding {}/{}[{}]={}".format(h5_filepath, group, attribute_name, attribute_value))
        if dry_run:
            continue
        h5_file = h5py.File(h5_filepath, 'a')
        dset = h5_file["{}/".format(group)]
        dset.attrs[attribute_name] = attribute_value
        h5_file.close()

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("--path", help="Can be either 1: path to individual hdf file, " +
        "2: txt file with list of hdf files, or " +
        "3: directory (all hdf files in directory will be processed).", required=True)
    required.add_argument("--attr_name", help="Name of new attribute", required=True)
    required.add_argument("--attr_val", help="Value of new attribute", required=True)
    optional.add_argument("--group", help="Group to add attribute to. Subgroups " +
            "are represented like paths, eg: /group1/subgroup2...", default="")
    optional.add_argument("--dry_run", default=0, type=int,
            help="If set to 1, will print changes without performing them")

    args = parser.parse_args()
    path = args.path
    extensions = [".hdf", ".h5"]
    files = get_filepaths_from_path_or_file(path, extensions=extensions)
    print(files)
    dry_run = False if args.dry_run <= 0 else True
    add_attribute(files, args.group, args.attr_name, args.attr_val, dry_run=dry_run)
