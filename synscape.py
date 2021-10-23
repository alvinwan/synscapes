"""Synscape dataloading utilities.

You may use this file directly as a conversion script, to convert
from .exr depth maps to .npy depth maps

    python synscape.py path/to/synscapes/directory/to/convert

Alternatively, inherit from these dataloaders in your own dataloader.

    from synscape import Synscape
    class MyDataset(Synscape):
        def __getitem__(self, index):
            paths = self.paths[index]
            image = self.load_image(paths['image'])
            depth = self.load_depth(paths['depth'])
            label = self.load_depth(paths['class'])
            instance = self.load_instance(paths['instance'])
            meta = self.load_meta(paths['meta'])

"""

import OpenEXR, Imath, os, sys, json, glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class Synscape(Dataset):

    def __init__(self, root):
        self.root = root
        self.paths = [
            {
                'image': os.path.join(root, f"img/rgb/{d}.png"),
                'depth': os.path.join(root, f"img/depth/{d}.exr"),
                'class': os.path.join(root, f"img/class/{d}.png"),
                'instance': os.path.join(root, f"img/instance/{d}.png"),
                'meta': os.path.join(root, f"meta/{d}.json"),
            }
        ]

    @staticmethod
    def load_depth_map(path, pt=Imath.PixelType(Imath.PixelType.FLOAT)):
        inp = OpenEXR.InputFile(path)
        dw = inp.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        buffer = inp.channel('Z', pt)
        depth = np.frombuffer(buffer, dtype=np.float32)
        depth.shape = (size[1], size[0]) # Numpy arrays are (row, col)
        depth = depth.copy() # current array is readonly
        depth[depth > 1000] = -1 # use -1 for 'ignore'
        return depth

    @staticmethod
    def load_json(path):
        with open(path) as f:
            return json.loads(path)

    @staticmethod
    def load_image(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def load_class(path, to_cityscapes=False):
        return Image.open(path)

    @staticmethod
    def load_instance(path):
        return Image.open(path)


class SynscapeDepth(Synscape):
    def __getitem__(self, index):
        paths = self.paths[index]

        image = self.load_image(paths['image'])
        depth = self.load_depth_map(paths['depth'])

        return image, depth


if __name__ == '__main__':
    root = sys.argv[1]
    for path in glob.iglob(f"{root}/img/depth/*.exr"):
        name = Path(path).stem
        np.save(f"{root}/depth_numpy/{name}.npy", Synscape.load_depth_map(path))