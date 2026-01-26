import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ScrewDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, samples_per_patient=1):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata for this split.
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            samples_per_patient (int): Number of random pairs to generate per patient per epoch.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples_per_patient = samples_per_patient
        
        # Group by patient_number
        self.patient_groups = df.groupby('patient_number')
        self.patient_ids = list(self.patient_groups.groups.keys())
        
        # Create a mapping from manufacturer name to class index
        self.manufacturers = sorted(df['manufacturer'].unique())
        self.class_to_idx = {m: i for i, m in enumerate(self.manufacturers)}
        self.idx_to_class = {i: m for m, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.patient_ids) * self.samples_per_patient

    def get_screw_crop(self, image_path, mask_path):
        """
        Loads image and mask, finds contours of screws, picks one random screw, and returns the crop.
        Returns a 'super-crop' with 2.0x context to allow for rotation without black borders.
        """
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            return None

        # Load image (Grayscale for X-ray model)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter small noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if not valid_contours:
            return None
            
        # Pick one random screw contour
        cnt = random.choice(valid_contours)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate center with slight jitter
        center_x = x + w / 2
        center_y = y + h / 2
        jitter_x = random.randint(-5, 5)
        jitter_y = random.randint(-5, 5)
        
        cx = center_x + jitter_x
        cy = center_y + jitter_y
        
        # Determine crop size (2.0x max dimension for context)
        max_dim = max(w, h)
        crop_size = int(max_dim * 2.0)
        half_size = crop_size // 2
        
        # Calculate coordinates
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Pad image if necessary to handle out-of-bounds
        h_img, w_img = image.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w_img)
        pad_b = max(0, y2 - h_img)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            # Adjust coordinates after padding
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t
            
        crop = image[y1:y2, x1:x2]
        
        return Image.fromarray(crop)

    def __getitem__(self, idx):
        # Map virtual index to actual patient index
        patient_idx = idx // self.samples_per_patient
        patient_id = self.patient_ids[patient_idx]
        group = self.patient_groups.get_group(patient_id)
        
        # Get manufacturer (should be same for all rows of a patient)
        manufacturer = group['manufacturer'].iloc[0]
        label = self.class_to_idx[manufacturer]
        
        # Separate views
        ap_images = group[group['view_position'] == 'AP']
        lat_images = group[group['view_position'] == 'LATERAL']
        
        if ap_images.empty or lat_images.empty:
            # Skip if missing a view.
            # In PyTorch DataLoader, returning None and using a custom collate_fn is the best way to handle this.
            return None 
            
        # Try to get valid crops (loop a few times in case of empty masks or bad files)
        max_attempts = 5
        
        ap_crop = None
        lat_crop = None
        
        # Get AP crop
        for _ in range(max_attempts):
            row = ap_images.sample(1).iloc[0]
            img_rel_path = row['relative_file_path']
            # Construct mask path: '.../images/filename.jpg' -> '.../masks/filename_mask.jpg'
            # Based on user description: "identically named file in the masks folder for that patient" 
            # and verify step showed '.../masks/109756947_00_mask.jpg'
            
            # The CSV relative_file_path is like: Depuy expedium (Synthes)/109756947/images/109756947_00.jpg
            img_path = os.path.join(self.root_dir, img_rel_path)
            
            # Mask path construction logic
            # Split path to replace 'images' with 'masks' and modify filename
            parts = img_rel_path.split('/')
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'masks'
                filename = parts[-1]
                name, ext = os.path.splitext(filename)
                parts[-1] = f"{name}_mask{ext}"
                mask_rel_path = "/".join(parts)
                mask_path = os.path.join(self.root_dir, mask_rel_path)
                
                ap_crop = self.get_screw_crop(img_path, mask_path)
                if ap_crop is not None:
                    break
        
        # Get Lateral crop
        for _ in range(max_attempts):
            row = lat_images.sample(1).iloc[0]
            img_rel_path = row['relative_file_path']
            img_path = os.path.join(self.root_dir, img_rel_path)
            
            parts = img_rel_path.split('/')
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'masks'
                filename = parts[-1]
                name, ext = os.path.splitext(filename)
                parts[-1] = f"{name}_mask{ext}"
                mask_rel_path = "/".join(parts)
                mask_path = os.path.join(self.root_dir, mask_rel_path)
                
                lat_crop = self.get_screw_crop(img_path, mask_path)
                if lat_crop is not None:
                    break

        if ap_crop is None or lat_crop is None:
            return None

        if self.transform:
            ap_crop = self.transform(ap_crop)
            lat_crop = self.transform(lat_crop)
            
        return ap_crop, lat_crop, label

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
