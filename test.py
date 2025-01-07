"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from models.networks import MGM
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
mav_value = 255
train_batch_size = 1
num_workers = 0


def change(x):
    y = x.cpu()
    return y.numpy()


def test(pred_cloudfree_data, i):
    save1_dir = 'C://Users//admin//Desktop//he//Dataset//test//opt_clear'

    pred_cloudfree_data1 = change(pred_cloudfree_data)
    a = pred_cloudfree_data1[0]
    # a=a.astype(np.)
    a = np.transpose(a, (1, 2, 0))
    a = (a * 255).astype(np.uint8)
    #  print(a)
    image = Image.fromarray(a)
    image.save(save1_dir + '//{}.png'.format(i))
img_dir_opt = 'C://Users//admin//Desktop//he//Dataset//test//opt_cloudy'
img_dir_vv = 'C://Users//admin//Desktop//he//Dataset//test//SAR//VV'
img_dir_vh = 'C://Users//admin//Desktop//he//Dataset//test//SAR//VH'


def load_image(filepath):  # 输入数据
    img = plt.imread(filepath)

    #  img = img.astype(np.float32)   # 归一化处理
    return img


def opt_xu(folder_path):  # 输入数据
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 定义一个正则表达式来提取a和b
    pattern = re.compile(r'ROIs_(\d{2})_p_(\d{4})\.png')

    # 对文件进行排序
    sorted_files = sorted(files, key=lambda x: (int(pattern.search(x).group(1)), int(pattern.search(x).group(2))))
    return sorted_files


def vv_xu(folder_path):  # 输入数据
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 定义一个正则表达式来提取a和b
    pattern = re.compile(r'ROIs_(\d{2})_VV_p_(\d{4})\.png')

    # 对文件进行排序
    sorted_files = sorted(files, key=lambda x: (int(pattern.search(x).group(1)), int(pattern.search(x).group(2))))
    return sorted_files


def vh_xu(folder_path):  # 输入数据
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 定义一个正则表达式来提取a和b
    pattern = re.compile(r'ROIs_(\d{2})_VH_p_(\d{4})\.png')

    # 对文件进行排序
    sorted_files = sorted(files, key=lambda x: (int(pattern.search(x).group(1)), int(pattern.search(x).group(2))))
    return sorted_files


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#


class DatasetFromFolder(Dataset):  # 数据预加载
    def __init__(self, img_dir_opt, img_dir_vv, img_dir_vh, transform=None):
        self.img_dir_opt = img_dir_opt
        self.input_file1 = opt_xu(img_dir_opt)
        self.img_dir_vv = img_dir_vv
        self.input_file2 = vv_xu(img_dir_vv)
        self.img_dir_vh = img_dir_vh
        self.input_file3 = vh_xu(img_dir_vh)

        self.transform = transform

    def __len__(self):  # 返回数据集的长度
        return len(self.input_file1)

    def __getitem__(self, index):  # idx的范围是从0到len（self）根据下标获取其中的一条数据
        input_opt_path = os.path.join(self.img_dir_opt, self.input_file1[index])

        input_opt = load_image(input_opt_path)

        input_vv_path = os.path.join(self.img_dir_vv, self.input_file2[index])

        input_vv = load_image(input_vv_path)

        input_vh_path = os.path.join(self.img_dir_vh, self.input_file3[index])

        input_vh = load_image(input_vh_path)

        if self.transform:
            input_opt = self.transform(input_opt)

            input_vv = self.transform(input_vv)

            input_vh = self.transform(input_vh)

        return input_opt, input_vv, input_vh


## 变换ToTensor
class ToTensor(object):
    def __call__(self, input):
        # 因为torch.Tensor的高维表示是 C*H*W,所以在下面执行from_numpy之前，要先做shape变换
        # 把H*W*C转换成 C*H*W  4*128*128
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def get_train_set(img_dir_opt, img_dir_vv, img_dir_vh):
    return DatasetFromFolder(img_dir_opt, img_dir_vv, img_dir_vh,
                             transform=transforms.Compose([ToTensor()]))

transformed_trainset = get_train_set(img_dir_opt,img_dir_vv,img_dir_vh)

test_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model2 = MGM(3,3)
    # model2.load_state_dict(torch.load('checkpoints/MGM_noquzao_29/latest_net_G_A.pth'))

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    prefetcher = DataPrefetcher(train_dataloader)
    data = prefetcher.next()
    i = 0

    while data is not None:
        i += 1
        if i >= 200:
            break
        opt = data[0].cuda()
        # print(data['cloudfree_data'].shape)
        vv = data[1].cuda()
        # print(vv.shape)
        vh = data[2].cuda()
        ref = data[3].cuda()
        size_average = True
        sar = torch.cat([vv[:, 0:1, :, :], vh[:, 0:1, :, :], vh[:, 0:1, :, :]], dim=1)  # inner loop within one epoch
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        test(pred_cloudfree_data, i)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
