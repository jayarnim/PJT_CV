import os
from tqdm import tqdm
import numpy as np
import torch
import cv2


def _video_to_frame(path, size):
    cap = cv2.VideoCapture(path)
    
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    return frames


def _frame_sampling(frames, num_frames):
    total = len(frames)
    
    if total >= num_frames:
        idx = np.linspace(0, total - 1, num_frames).astype(int)
        frames = [frames[i] for i in idx]
    else:
        pad = [frames[-1]] * (num_frames - total)
        frames += pad

    frames = np.stack(frames)  # (T, H, W, C)
    frames = frames.transpose(3, 0, 1, 2)  # (C, T, H, W)

    return frames


def load_video(path, num_frames=32, size=224):
    frames = _video_to_frame(path, size)
    samples = _frame_sampling(frames, num_frames)
    slow = samples[:, ::4, :, :]
    fast = samples
    return [slow / 255.0, fast / 255.0]


def cache_videos(
    source_dir: str,  # ex: "./RWF-2000/data/train"
    target_dir: str,  # ex: "./RWF-2000/cached/train"
    num_frames: int=32,
    size: int=224,
    extension: str=".avi"
):
    os.makedirs(target_dir, exist_ok=True)

    for label_name in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label_name)
        
        # 경로가 유효하지 않을 경우 생략
        if not os.path.isdir(label_path):
            continue

        save_dir = os.path.join(target_dir, label_name)
        os.makedirs(save_dir, exist_ok=True)

        iter_obj = tqdm(
            iterable=os.listdir(label_path),
            desc=f"Caching {label_name}",
        )

        for fname in iter_obj:
            # 확장자가 조건을 충족하지 않을 경우 생략
            if not fname.endswith(extension):
                continue

            video_path = os.path.join(label_path, fname)
            save_path = os.path.join(save_dir, fname.replace(extension, ".pt"))

            # 이미 저장된 경우 생략
            if os.path.exists(save_path):
                continue

            slow, fast = load_video(video_path, num_frames=num_frames, size=size)
            slow = torch.tensor(slow, dtype=torch.float32)
            fast = torch.tensor(fast, dtype=torch.float32)
            torch.save([slow, fast], save_path)

    print("✅ Done caching videos.")
