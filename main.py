import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO

# Import and concatenate MOD1
mod1_parts = []
for i in range(1, 9):
    temp = pd.read_csv(f"data/Libelium New/part{i}/mod1.txt", sep="\t", header=None, names=("Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620"))
    mod1_parts.append(temp)

mod1 = pd.concat(mod1_parts)
print("MOD 1 data:")
print(mod1.head())

# Import and concatenate MOD2
mod2_parts = []
for i in range(1, 9):
    temp = pd.read_csv(f"data/Libelium New/part{i}/mod2.txt", sep="\t", header=None, names=("Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620"))
    mod2_parts.append(temp)

mod2 = pd.concat(mod2_parts)
print("MOD 2 data:")
print(mod2.head())

# Import and concatenate PODs
pod_files = {
    "POD 200085": ["data/PODs/14_nov-22_nov-Pods/POD 200085.csv", "data/PODs/23_nov-12_dec-Pods/POD 200085.csv", "data/PODs/fevrier_mars_2023_pods/POD 200085.csv"],
    "POD 200086": ["data/PODs/14_nov-22_nov-Pods/POD 200086.csv", "data/PODs/23_nov-12_dec-Pods/POD 200086.csv", "data/PODs/fevrier_mars_2023_pods/POD 200086.csv"],
    "POD 200088": ["data/PODs/14_nov-22_nov-Pods/POD 200088.csv", "data/PODs/23_nov-12_dec-Pods/POD 200088.csv", "data/PODs/fevrier_mars_2023_pods/POD 200088.csv"]
}

pods = {}

for key, value in pod_files.items():
    pod_parts = []
    for file_path in value:
        temp = pd.read_csv(file_path, sep=";", skiprows=(1, 2, 3, 4))
        pod_parts.append(temp)
    pods[key] = pd.concat(pod_parts)
    print(f"{key} data:")
    print(pods[key].head())

# PICO, Piano THICK and Piano THIN are similar to the POD modules, use the same code but provide appropriate file paths
