"""
Explore sample data of INSTANCE data set.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import h5py


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument(
        "--input_dir",
        default="/home/mila/h/hernanga/scratch/datasets/earthquakes",
        type=str,
        help="Path to directory containing the data. It should contain a ./metadata "
        "and a ./data directory.",
    )
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def read_data(datapath):
    """
    Reads data from a directory containing the metadata CSV files in ./metadata and the
    data HDF5 files in ./data.

    Returns
    -------
    Tuple :
        A tuple of 3 tuples with 2 components (data, metadata) each:
         - (h5_events_counts, df_events_metadata)
         - (h5_events_gm, df_events_metadata)
         - (h5_noise, df_noise_metadata)
    """
    datapath = Path(datapath)
    # Read events metadata
    events_filename = datapath / "metadata" / "metadata_Instance_events_10k.csv"
    df_events_metadata = pd.read_csv(
        events_filename,
        keep_default_na=False,
        dtype={
            "station_location_code": object,
            "source_mt_eval_mode": object,
            "source_mt_status": object,
            "source_mechanism_strike_dip_rake": object,
            "source_mechanism_moment_tensor": object,
            "trace_p_arrival_time": object,
            "trace_s_arrival_time": object,
        },
        low_memory=False,
    )
    # Read noise metadata
    noise_filename = datapath / "metadata" / "metadata_Instance_noise_1k.csv"
    df_noise_metadata = pd.read_csv(
        noise_filename, dtype={"station_location_code": object}, low_memory=False
    )
    # Read events counts HDF5
    events_counts_filename = datapath / "data" / "Instance_events_counts_10k.hdf5"
    h5_events_counts = h5py.File(events_counts_filename, "r")
    # Read events GM HDF5
    events_gm_filename = datapath / "data" / "Instance_events_gm_10k.hdf5"
    h5_events_gm = h5py.File(events_gm_filename, "r")
    # Read noise HDF5
    noise_filename = datapath / "data" / "Instance_noise_1k.hdf5"
    h5_noise = h5py.File(noise_filename, "r")
    return (
        (h5_events_counts, df_events_metadata),
        (h5_events_gm, df_events_metadata),
        (h5_noise, df_noise_metadata),
    )


def main(args):
    (
        (h5_events_counts, df_events_metadata),
        (h5_events_gm, df_events_metadata),
        (h5_noise, df_noise_metadata),
    ) = read_data(args.input_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    sys.exit()
