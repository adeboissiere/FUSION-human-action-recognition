"""
The main file for the *src.data* module. Creates an h5 dataset from the raw NTU RGB+D files.
Best called using the Makefile provided.

>>> make data \\
    RAW_DATA_PATH=X \\
    PROCESSED_DATA_PATH=X \\
    DATASET_TYPE=X \\
    COMPRESSION=X \\
    COMPRESSION_OPTS=X

With the parameters taking from the following values :
    - RAW_DATA_PATH:
        Default value is *./data/raw/*
    - PROCESSED_DATA_PATH:
        Default value is *./data/processed/*
    - DATASET_TYPE:
        [SKELETON | IR_SKELETON | IR | IR_CROPPED | IR_CROPPED_MOVING]
    - COMPRESSION:
        ["", lzf, gzip]
    - COMPRESSION_OPTS (gzip compression only):
        [1, .., 9]

"""

import argparse
from src.data.make_dataset_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU RGB-D feature extraction')
    parser.add_argument('--data_path')
    parser.add_argument('--output_folder')
    parser.add_argument('--dataset_type')
    parser.add_argument('--compression')
    parser.add_argument('--compression_opts', default=9)

    arg = parser.parse_args()

    print("\r\n\r\n===== CREATING H5 DATASET =====")
    print("-> NTU RGB-D dataset path :" + str(arg.data_path))
    print("-> Output folder path : " + str(arg.output_folder))
    print("-> Dataset type : " + str(arg.dataset_type))

    if arg.compression != "":
        print("-> Compression type : " + str(arg.compression))

        if arg.compression == "gzif":
            print("Compression opts : " + str(arg.compression_opts))

    if arg.dataset_type == "SKELETON":
        create_h5_skeleton_dataset(arg.data_path,
                                   arg.output_folder,
                                   arg.compression,
                                   arg.compression_opts)

    elif arg.dataset_type == "IR_SKELETON":
        create_h5_2d_ir_skeleton(arg.data_path,
                                 arg.output_folder,
                                 arg.compression,
                                 arg.compression_opts)

    elif arg.dataset_type == "IR":
        create_h5_ir_dataset(arg.data_path,
                             arg.output_folder,
                             arg.compression,
                             arg.compression_opts)

    elif arg.dataset_type == "IR_CROPPED":
        create_h5_ir_cropped_dataset_from_h5(arg.data_path,
                                             arg.output_folder,
                                             arg.compression,
                                             arg.compression_opts)

    elif arg.dataset_type == "IR_CROPPED_MOVING":
        create_h5_ir_cropped_moving_dataset_from_h5(arg.data_path,
                                                    arg.output_folder,
                                                    arg.compression,
                                                    arg.compression_opts)

    else:
        print("Data set type not recognized")
