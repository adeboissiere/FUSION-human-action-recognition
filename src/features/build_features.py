import argparse
from src.utils.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU RGB-D feature extraction')
    parser.add_argument(
        '--data_path', default="/media/gnocchi/Seagate Backup Plus Drive/NTU-RGB-D/"
    )
    parser.add_argument(
        '--output_folder', default="/media/gnocchi/Seagate Backup Plus Drive/NTU-RGB-D/"
    )
    parser.add_argument('--dataset_type', default="")
    parser.add_argument('--compression', default="")
    parser.add_argument('--compression_opts', default=9)

    arg = parser.parse_args()

    print("===== CREATING H5 DATASET =====")
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
    elif arg.dataset_type == "IR":
        create_h5_ir_dataset(arg.data_path,
                             arg.output_folder,
                             arg.compression,
                             arg.compression_opts)
    else:
        print("Data set type not recognized")
