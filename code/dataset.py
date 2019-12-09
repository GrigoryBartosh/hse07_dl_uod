import os

from PIL import Image
from torch.utils.data import Dataset


class EmojiDataset(Dataset):

    def __init__(self, root, transform):
        self.root_dir = root
        self.transform = transform
        self.len = len(os.listdir(self.root_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(idx) + '.png')
        sample = Image.open(img_name)

        sample = self.transform(sample)[:3]

        return sample
