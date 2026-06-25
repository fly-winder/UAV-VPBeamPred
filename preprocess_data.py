# preprocess_data.py
import os
import pandas as pd
import numpy as np
import ast
import random

csv_file = "scenario23_dev/scenario23.csv"
img_dir = "scenario23_dev"
T_in = 8
T_out = 5

df = pd.read_csv(csv_file)



seq_list = sorted(df['seq_index'].unique())

# seq_list = list(df['seq_index'].unique())
# random.shuffle(seq_list)  



def read_txt(path):
    full_path = os.path.join(img_dir, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found")
    arr = []
    with open(full_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            arr.append([float(x) for x in parts])
    return arr



all_lats, all_lons, all_distances, all_heights = [], [], [], []

for idx in range(len(df)):
    loc_arr = read_txt(df.iloc[idx]['unit2_loc'])
    all_lats.append(loc_arr[0][0])
    all_lons.append(loc_arr[1][0])

    distance_arr = read_txt(df.iloc[idx]['unit2_distance'])
    all_distances.append(distance_arr[0][0])

    height_arr = read_txt(df.iloc[idx]['unit2_height'])
    all_heights.append(height_arr[0][0])

lat_min, lat_max = min(all_lats), max(all_lats)
lon_min, lon_max = min(all_lons), max(all_lons)
dist_min, dist_max = min(all_distances), max(all_distances)
height_min, height_max = min(all_heights), max(all_heights)










def normalize(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin + 1e-8)  




def create_window_rows(df_group, T_in=8, T_out=5):
    df_group = df_group.sort_values('index')
    rows = []
    for start in range(len(df_group) - T_in - T_out + 1):
        in_idx = slice(start, start + T_in)
        out_idx = slice(start + T_in, start + T_in + T_out)

        # Define the index of the current time step, i.e., the last frame of the input sequence
        current_idx = start + T_in - 1
        row = {}

        # Historical 8-frame image paths
        rgb_list = [
            os.path.join(img_dir, df_group.iloc[i]['unit1_rgb'].lstrip("./")).replace("\\", "/")
            for i in range(in_idx.start, in_idx.stop)
        ]
        row['unit1_rgb'] = rgb_list  # list of str

        # Historical 8-frame position coordinates (8, 2)
        loc_list = []
        for i in range(in_idx.start, in_idx.stop):
            loc_arr = read_txt(df_group.iloc[i]['unit2_loc'])
            lat = normalize(loc_arr[0][0], lat_min, lat_max)
            lon = normalize(loc_arr[1][0], lon_min, lon_max)
            loc_list.append([lat, lon])
        row['unit2_loc'] = loc_list

        # Historical 8-frame speed values (8, 1)
        speed_list = []
        for i in range(in_idx.start, in_idx.stop):
            speed_arr = read_txt(df_group.iloc[i]['unit2_speed'])
            speed_list.append([speed_arr[0][0]])
        row['unit2_speed'] = speed_list

        # Historical 8-frame distance values (8, 1)
        distance_list = []
        for i in range(in_idx.start, in_idx.stop):
            distance_arr = read_txt(df_group.iloc[i]['unit2_distance'])
            # distance_list.append([distance_arr[0][0]])
            distance_list.append([normalize(distance_arr[0][0], dist_min, dist_max)])
        row['unit2_distance'] = distance_list

        # Historical 8-frame height values (8, 1)
        height_list = []
        for i in range(in_idx.start, in_idx.stop):
            height_arr = read_txt(df_group.iloc[i]['unit2_height'])
            # height_list.append([height_arr[0][0]])
            height_list.append([normalize(height_arr[0][0], height_min, height_max)])
        row['unit2_height'] = height_list

        # Future 5-frame beam indices (5, 1)
        beam_list = []
        for i in range(out_idx.start, out_idx.stop):
            beam_list.append([int(df_group.iloc[i]['unit1_beam_index'])])
        row['unit1_beam_index'] = beam_list

        # Future 5-frame codebook power distributions (5, 64)
        pwr_list = []
        for i in range(out_idx.start, out_idx.stop):
            # Read the file. The content format is assumed to be [[0.02], [0.025], ...]
            # Here, read_txt is reused, assuming it can process the path and return a list structure
            pwr_raw = read_txt(df_group.iloc[i]['unit1_pwr_60ghz'])

            # Data cleaning: flatten [[x], [y], ...] into [x, y, ...]
            # Each time step finally obtains a list with length 64
            if pwr_raw and isinstance(pwr_raw[0], list):
                pwr_flat = [item[0] for item in pwr_raw]
            else:
                # Compatibility handling: used when read_txt has already flattened the data or the format is different
                pwr_flat = pwr_raw

            pwr_list.append(pwr_flat)
        row['unit1_pwr_60ghz'] = pwr_list

        # ==================================================
        # [New] Obtain the raw physical quantities at the current time step
        # i.e., the last frame of the input sequence
        # These values are used for subsequent performance analysis
        # and are stored as non-normalized values
        # ==================================================

        # 1. Current speed
        curr_speed_raw = read_txt(df_group.iloc[current_idx]['unit2_speed'])
        row['current_speed'] = curr_speed_raw[0][0]  # stores a float

        # 2. Current distance
        curr_dist_raw = read_txt(df_group.iloc[current_idx]['unit2_distance'])
        row['current_distance'] = curr_dist_raw[0][0]  # stores a float, not normalized

        # 3. Current height
        curr_height_raw = read_txt(df_group.iloc[current_idx]['unit2_height'])
        row['current_height'] = curr_height_raw[0][0]  # stores a float, not normalized

        rows.append(row)

    return rows



def build_split(df, seq_subset):
    all_rows = []
    for seq_id in seq_subset:
        group = df[df['seq_index'] == seq_id]
        rows = create_window_rows(group, T_in, T_out)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)



seq_list = sorted(df['seq_index'].unique())


data_df = build_split(df, seq_list)

test_df = data_df.sample(frac=0.25, random_state=42)

train_df = data_df.drop(test_df.index)

data_df.to_csv("data_windows_all.csv", index=False)
train_df.to_csv("data_windows.csv", index=False)
test_df.to_csv("test_windows.csv", index=False)

print(f"Total samples: {len(data_df)}")
