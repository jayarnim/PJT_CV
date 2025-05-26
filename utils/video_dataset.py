import os
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        label_map: dict, 
        extension: str=".pt",
    ):
        # 루트 폴더 경로
        self.root_dir = root_dir
        # 폴더 이름 to 레이블 ex) {'Fight': 1, 'NonFight': 0}
        self.label_map = label_map
        # 확장자
        self.extension = extension

        self._collect()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = torch.load(video_path)
        label = self.labels[idx]
        return frames, label

    def _collect(self):
        self.video_paths = []
        self.labels = []
        
        for label_name, label in self.label_map.items():
            label_dir = os.path.join(self.root_dir, label_name)
            
            for fname in os.listdir(label_dir):
                if fname.endswith(self.extension):
                    self.video_paths.append(os.path.join(label_dir, fname))
                    self.labels.append(float(label))
