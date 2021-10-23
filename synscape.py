import OpenEXR, Imath, os
from PIL import Image
import numpy as np
import json
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