"""
Drive and act Dataloader

"""


import imageio
from PIL import Image
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import  Dataset
from sklearn.preprocessing import LabelEncoder




from utils.video_slicer import slice_frame
class DriveAndAct(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.dataset = config.data_path

        conv = {1: lambda x: int(x), 2: lambda x: int(x), 4: lambda x: int(x)}
        data = np.loadtxt(config.label_path + "distracted_driving.csv", dtype=str, delimiter=",", skiprows=1,
                          converters=conv)
        self.X = data[:, :3]
        self.y = data[:, 3]
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)


    def __getitem__(self, index: int):
        sample = self.X[index]
        video = slice_frame(self.dataset,
                            {'file_id': sample[0], 'frame_start': sample[1], 'frame_end': sample[2]})
        video = [Image.fromarray(np.uint8(frame * 255)).resize((224, 224)) for frame in video]
        label = self.y[index]
        # if self.transform is not None:
        #     video = self.transform(np.array(video))

        return video, label

    def __len__(self) -> int:
        return len(self.X)