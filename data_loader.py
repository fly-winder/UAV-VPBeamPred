"""
8-frame images (B, 8, 3, H, W)
8-frame positions (B, 8, 2)
5-frame future beam labels (B, 5)
"""

import os
import ast
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

class BeamDataset(Dataset):
    def __init__(self, csv_file, root_dir="", transform=None):
        """
        Args:
            csv_file (str): Path to the preprocessed CSV file
            root_dir (str): Root directory of the dataset
            transform: Image transform
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---------- Images ----------
        img_paths = ast.literal_eval(row["unit1_rgb"]) 
        imgs = []
        for p in img_paths:
            img_path = os.path.join(self.root_dir, p)  
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image) 
            imgs.append(image)
        imgs = torch.stack(imgs) 

        # ---------- Positions (8,2) ----------
        loc = torch.tensor(ast.literal_eval(row["unit2_loc"]), dtype=torch.float32)

        # ---------- speed/distance/height (8,1) ----------
        # speed = torch.tensor(ast.literal_eval(row["unit2_speed"]), dtype=torch.float32)
        distance = torch.tensor(ast.literal_eval(row["unit2_distance"]), dtype=torch.float32)
        height = torch.tensor(ast.literal_eval(row["unit2_height"]), dtype=torch.float32)

        # ---------- beam_index (5,1) ----------
        beam = torch.tensor(ast.literal_eval(row["unit1_beam_index"]), dtype=torch.long).squeeze(-1)

        sample = {
            "images": imgs,  # (8, C, H, W)
            "loc": loc,      # (8, 2)
            # "speed": speed,  # (8, 1)
            "distance": distance,  # (8, 1)
            "height": height,  # (8, 1)
            "beam": beam      # (5,)
        }
        return sample

