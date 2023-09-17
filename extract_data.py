import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_data():
    df = pd.read_csv("event.csv")
    trace_names = set(df["trace_name"])
    print(len(trace_names))

    # filename = "/home/mila/y/yuyan.chen/Earthquake-detection/data/Instance_noise_1k.hdf5"
    filename = (
        "/home/mila/y/yuyan.chen/Earthquake-detection/data/Instance_events_gm_10k.hdf5"
    )
    file = h5py.File(filename, "r")

    dataset = np.array([np.array(file["data"][tr]) for tr in trace_names])
    print(dataset.shape)
    np.save(f"events_gm.npy", dataset)


def visualize():
    var_name = "events_gm"
    filename = (
        f"/home/mila/y/yuyan.chen/Earthquake-detection/extracted_data/{var_name}.npy"
    )
    data = np.load(filename)
    plt.plot(data[0][0][:2000])
    plt.title(var_name)
    plt.savefig(f"{var_name}_2000.png")


visualize()
