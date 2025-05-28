import os
import random
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img


class PromptTrainDataset(Dataset):
    def __init__(self, args, part='train'):
        super().__init__()
        self.args = args
        self.part = part

        self.deg_dir = os.path.join(args.derain_dir, part, 'degraded')
        self.clean_dir = os.path.join(args.derain_dir, part, 'clean')

        all_f = sorted(os.listdir(self.deg_dir))
        self.names = [f for f in all_f if f.lower().endswith('.png')]
        if part == 'train':
            random.shuffle(self.names)

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        fn = self.names[idx]
        deg_path = os.path.join(self.deg_dir, fn)
        clean_fn = (
            fn
            .replace('rain-', 'rain_clean-', 1)
            .replace('snow-', 'snow_clean-', 1)
        )
        clean_path = os.path.join(self.clean_dir, clean_fn)

        deg_img = crop_img(
            np.array(Image.open(deg_path).convert('RGB')),
            base=16
        )

        clean_img = crop_img(
            np.array(Image.open(clean_path).convert('RGB')),
            base=16
        )

        deg_patch, clean_patch = self._crop_patch(deg_img, clean_img)
        deg_patch, clean_patch = random_augmentation(deg_patch, clean_patch)

        deg_t = self.to_tensor(deg_patch)
        clean_t = self.to_tensor(clean_patch)
        return deg_t, clean_t

    def _crop_patch(self, im1, im2):
        H, W, _ = im1.shape
        ps = self.args.patch_size
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        p1 = im1[top:top+ps, left:left+ps, :]
        p2 = im2[top:top+ps, left:left+ps, :]
        return p1, p2


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = (
            np.clip(clean_patch + noise * self.sigma, 0, 255)
            .astype(np.uint8)
        )

        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(
            np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')),
            base=16
        )

        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img = self.toTensor(clean_img)
        noisy_img = self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img

    def tile_degrad(input_, tile=128, tile_overlap=0):
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[
                    ...,
                    h_idx:(h_idx+tile),
                    w_idx:(w_idx+tile)
                ].add_(out_patch)

                W[
                    ...,
                    h_idx:(h_idx+tile),
                    w_idx:(w_idx+tile)
                ].add_(out_patch_mask)
        restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored

    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain", addnoise=False, sigma=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = (
            np.clip(clean_patch + noise * self.sigma, 0, 255)
            .astype(np.uint8)
        )

        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            input_derain_dir = self.args.derain_path + 'input/'
            self.ids += [input_derain_dir + id_ for id_ in name_list]

        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            input_dehaze_dir = self.args.dehaze_path + 'input/'
            self.ids += [input_dehaze_dir + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(
            np.array(Image.open(degraded_path).convert('RGB')),
            base=16
        )

        if self.addnoise:
            degraded_img, _ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(
            np.array(Image.open(clean_path).convert('RGB')),
            base=16
        )

        clean_img = self.toTensor(clean_img)
        degraded_img = self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                msg = "The input directory does not contain any image files"
                raise Exception(msg)
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(
            np.array(Image.open(self.degraded_ids[idx]).convert('RGB')),
            base=16
        )

        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
