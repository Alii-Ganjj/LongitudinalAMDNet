import os
import logging
import argparse
import torch
import random
import numpy as np

from training import train_pix2pix
from torch.utils.tensorboard import SummaryWriter
from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import set_logging_settings, pair_visits
from Utils.utils import add_common_args
from Models.NextStepModels import Pix2PixModel


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int,  default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='1', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--gpu_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=2, help='loads 2 samples in train/val/test datasets.')
parser.add_argument('--num_save_epochs', default=5)
parser.add_argument('--num_save_image', default=5)
parser.add_argument('--num_test_iters', default=100)
parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
parser.add_argument('--num_test_epochs', default=1)
parser.add_argument('--batch_size_train', default=1)
parser.add_argument('--continue_train', default=False)
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by '
                                                               '<epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--test_mode_eval', default=False, help='controls batch norm and dropout in eval mode.')
parser.add_argument('--load_iter', default=0, help='Whether to load from iter. (Not important)')
parser.add_argument('--epoch', default=1)

# Data Parameters
parser.add_argument('--balanced', default=True, help='Whether to discard samples to balance the dataset or not.')
parser.add_argument('--transform', default=['resize'])
parser.add_argument('--im_resize_shape', default=224)
parser.add_argument('--visit_gap', default=4, help='Time duration (in units of x6 months) between visit pairs.')
parser.add_argument('--late_AMD_threshold', default=10, help='9/10. The threshold score for late AMD')
parser.add_argument('--min_visit_no', default=0, help='Minimum possible visit number for a subject.')
parser.add_argument('--max_visit_no', default=26, help='Maximum possible visit number for a subject.')
parser.add_argument('--min_participant_pairs', default=1, help='min # of pairs to use for each eye in each participant')

# Generator Parameters
parser.add_argument('--input_nc', default=3)
parser.add_argument('--output_nc', default=3)
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--norm', default='batch')
parser.add_argument('--net_G', default='unet_256', help='unet_256, unet_128')
parser.add_argument('--lambda_L1', default=100.)
parser.add_argument('--pool_size', default=0)
parser.add_argument('--gan_mode', default='vanilla', help='vanilla| lsgan | wgangp')

# Discriminator Parameters
parser.add_argument('--net_D', type=str, default='basic', help='[basic | n_layers | pixel]')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming '
                                                                    '| orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--no_dropout', default=False, help='no dropout for the generator')

# Optimizer Parameters
parser.add_argument('--lr', default=0.0002)
parser.add_argument('--beta1', default=0.5)
parser.add_argument('--beta2', default=0.999)
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')

parser = add_common_args(parser)
args = parser.parse_args()

# ############################# Logging & Fixing Seed #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if int(args.gpus) >= 0:
    torch.cuda.manual_seed_all(args.random_seed)

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])
args.writer = SummaryWriter(args.stdout)
# ############################# Loading Data #############################
if args.debug:
    args.epoch_count = 1
    args.n_epochs = 2
    args.n_epochs_decay = 0
    args.num_save_epochs = 1

args.visits = pair_visits(args.min_visit_no, args.max_visit_no, args.visit_gap)
Data = AMDDataAREDS(args)
Data.prepare_data()
Data.setup()
partitions = ['whole', 'train', 'val', 'test']
for p in partitions:
    logging.warning('************************')
    logging.warning('Histogram of ' + p + ' partition:')
    Data.data_histogram(partition=p, log=True)
    logging.warning('************************')

# ############################# Defining Model and Optimizer #############################
model_GAN = Pix2PixModel(args).to(args.device)
model_GAN.setup(args)
train_pix2pix(model_GAN, Data, args)
