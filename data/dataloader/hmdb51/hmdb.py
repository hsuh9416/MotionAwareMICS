import tensorflow_datasets as tdfs

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class HMDB51(VisionDataset):
    """HMDB51 Dataset."""
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # Load HMDB51 from Hugging Face
        self.ds = load_dataset("NoahMartinezXiang/HMDB51", split=split)
        # Decode video column into frame-level numpy array
        self.ds = self.ds.cast_column("video", Video())

        # augment / preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(config.resize),
            transforms.RandomCrop(config.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)])

        self.num_frames = config.num_frames

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        # numpy array (T, H, W, C)
        frames = example["video"]["array"]
        T = frames.shape[0]

        # Random or sequential frame sampling
        if T >= self.num_frames:
            start = random.randint(0, T - self.num_frames)
            clip = frames[start: start + self.num_frames]
        else:
            # In case of insufficient padding, repeat the last frame
            pad = self.num_frames - T
            last = frames[-1:]
            clip = np.concatenate([frames, np.repeat(last, pad, axis=0)], axis=0)

        # Apply transform to each frame
        clip = [
            self.transform(Image.fromarray(frame))
            for frame in clip
        ]
        # shape: (num_frames, C, H, W) -> (C, num_frames, H, W)
        clip = torch.stack(clip).permute(1, 0, 2, 3)

        label = example["label"]
        return clip, label

    def get_hmdb51_dataloader(self, args, split: str):
        dataset = HMDB51Dataset(split, args)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=config.drop_last,
        )
        return dataset, loader