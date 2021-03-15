"""
This script loads the 3 trained GAN model checkpoints and performs longitudinal prediction using them.
We obtained the figures in the paper by using the flag "--no_dropout=True" for reproducibility.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import set_logging_settings, pair_visits
from Utils.utils import add_common_args
from Models.NextStepModels import Pix2PixModel

from skimage import io
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int,  default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='0', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--gpu_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=5, help='loads 2 samples in train/val/test datasets.')
parser.add_argument('--num_save_epochs', default=20)
parser.add_argument('--num_save_image', default=5)
parser.add_argument('--num_test_iters', default=100)
parser.add_argument('--num_test_epochs', default=1)
parser.add_argument('--batch_size_train', default=1)
parser.add_argument('--continue_train', default=False)
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
parser.add_argument('--test_mode_eval', default=False, help='controls batch norm and dropout in eval mode.')
parser.add_argument('--load_iter', default=0, help='Whether to load from iter. (Not important)')
parser.add_argument('--epoch', default=30, help="checkpoint's epoch number.")
parser.add_argument('--num_prediction', default=320)

# Data Parameters
parser.add_argument('--visit_gap', default=4, help='Time duration (in units of x6 months) between visit pairs.')
parser.add_argument('--late_AMD_threshold', default=10, help='9/10. The threshold score for late AMD')
parser.add_argument('--min_visit_no', default=0, help='Minimum possible visit number for a subject.')
parser.add_argument('--max_visit_no', default=26, help='Maximum possible visit number for a subject.')
parser.add_argument('--min_participant_pairs', default=1, help='min # of pairs to use for each eye in each participant')
parser.add_argument('--transform', default=['resize'])
parser.add_argument('--im_resize_shape', default=224)
parser.add_argument('--binary', default=False)
parser.add_argument('--balanced', default=True)

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
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming '
                                                                    '| orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', default=True, help='no dropout for the generator')

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
    args.batch_size_train = 1
    args.n_epochs = 1
    args.n_epochs_decay = 0

args.visits = pair_visits(args.min_visit_no, args.max_visit_no, args.visit_gap)
Data = AMDDataAREDS(args)
Data.prepare_data()
Data.setup()
np.random.seed(args.random_seed)

partitions = ['whole', 'train', 'val', 'test']
for p in partitions:
    logging.warning('************************')
    logging.warning('Histogram of ' + p + 'partition:')
    Data.data_histogram(partition=p, log=True)
    logging.warning('************************')


def prepare_image_list(datamodule, partition):
    if partition == 'train':
        dataset = datamodule.train_set
    elif partition == 'val':
        dataset = datamodule.val_set
    elif partition == 'test':
        dataset = datamodule.test_set
    samples_image_names = []
    for sample_dict in dataset.samples_dict:
        sample_image_name_RE, sample_image_name_LE = [], []
        for pair in sample_dict['LE']:
            sample_image_name_LE.append(pair[0][0])
            sample_image_name_LE.append(pair[0][1])
        for pair in sample_dict['RE']:
            sample_image_name_RE.append(pair[0][0])
            sample_image_name_RE.append(pair[0][1])
        sample_image_name_LE = np.unique(np.array(sample_image_name_LE))
        sample_image_name_RE = np.unique(np.array(sample_image_name_RE))
        samples_image_names.append(sample_image_name_RE)
        samples_image_names.append(sample_image_name_LE)
    return samples_image_names


def prepare_0_4_6_8(seq):
    has_0, has_4, has_6, has_8 = False, False, False, False
    img_0, img_4, img_6, img_8 = None, None, None, None
    seq_0_4_6_8 = []
    for name in seq:
        img_name = (name.split('/'))[-1]
        visit_no = (img_name.split('_'))[1]
        if (visit_no.lower() == 'qua') or (visit_no.lower() == '00'):
            has_0 = True
            img_0 = name
        if visit_no.lower() == '04':
            has_4 = True
            img_4 = name
        if visit_no.lower() == '06':
            has_6 = True
            img_6 = name
        if visit_no.lower() == '08':
            has_8 = True
            img_8 = name
    if has_0 and has_4 and has_6 and has_8:
        seq_0_4_6_8.append(img_0)
        seq_0_4_6_8.append(img_4)
        seq_0_4_6_8.append(img_6)
        seq_0_4_6_8.append(img_8)
    return seq_0_4_6_8


train_seq, val_seq = prepare_image_list(Data, 'train'), prepare_image_list(Data, 'val')
test_seq = prepare_image_list(Data, 'test')

train_long_seq, val_long_seq, test_long_seq = [], [], []
for seq in train_seq:
    long_seq = prepare_0_4_6_8(seq)
    if long_seq:
        train_long_seq.append(long_seq)

for seq in val_seq:
    long_seq = prepare_0_4_6_8(seq)
    if long_seq:
        val_long_seq.append(long_seq)

for seq in test_seq:
    long_seq = prepare_0_4_6_8(seq)
    if long_seq:
        test_long_seq.append(long_seq)

# ############################# Loading Checkpoints #############################
checkpoint_dir_4 = './Datasets/checkpoints/GAN_4'
checkpoint_dir_6 = './Datasets/checkpoints/GAN_6'
checkpoint_dir_8 = './Datasets/checkpoints/GAN_8'

epoch_4 = 35
args.checkpoint_dir = checkpoint_dir_4
args.epoch = epoch_4
GAN_4 = Pix2PixModel(args, isTrain=False).to(args.device)
GAN_4.setup(args)
GAN_4.train()


epoch_6 = 40
args.checkpoint_dir = checkpoint_dir_6
args.epoch = epoch_6
GAN_6 = Pix2PixModel(args, isTrain=False).to(args.device)
GAN_6.setup(args)
GAN_6.train()


epoch_8 = 45
args.checkpoint_dir = checkpoint_dir_8
args.epoch = epoch_8
GAN_8 = Pix2PixModel(args, isTrain=False).to(args.device)
GAN_8.setup(args)
GAN_8.train()


def predict_next_images(model, data_loader, opt):
    model.train()
    generated_images = []
    y_data_loader = []
    if opt.test_mode_eval:
        model.eval()

    for i, sample in enumerate(data_loader, 0):
        x1, x2, y = sample[0][0].to(opt.device), sample[0][1].to(opt.device), sample[1].to(opt.device)
        model.set_input((x1, x2))
        model.test()
        generated_images.append(model.fake_B)
        y_data_loader.append(y)

    generated_images, y_data = torch.stack(generated_images, 0).squeeze(), torch.stack(y_data_loader, 0).squeeze()
    return generated_images, y_data


def load_image(im_dir, resize_shape):
    transform = transforms.Compose([transforms.Resize(resize_shape), transforms.ToTensor()])
    im = io.imread(im_dir)
    image = Image.fromarray(im)
    sample_image = transform(image)
    return sample_image


def load_seq_images(seq, sev_dict, resize_shape):
    images = []
    labels = []
    for name in seq:
        img = load_image(name, resize_shape)
        images.append(img)
        try:
            labels.append(sev_dict[(name.split('/'))[-1]])
        except:
            return [], []
    images = torch.stack(images, dim=0).squeeze()
    return images, labels


def tensor_to_image(im_tensor):
    im_np = im_tensor.cpu().numpy()
    image = Image.fromarray(im_np, 'RGB')
    return image


def make_prediction(curr_img, GAN_4step, GAN_6step, GAN_8step):
    GAN_4step.set_input((curr_img, curr_img))
    GAN_6step.set_input((curr_img, curr_img))
    GAN_8step.set_input((curr_img, curr_img))
    GAN_4step.test(), GAN_6step.test(), GAN_8step.test()
    pred_4, pred_6, pred_8 = GAN_4step.fake_B, GAN_6step.fake_B, GAN_8step.fake_B
    preds = torch.stack([pred_4, pred_6, pred_8], dim=0)
    return preds.squeeze()


def sev_scale_xlabel(sev_scales):
    x_label = 'Sev Scale: '
    x_label += str(sev_scales[0])
    num_sample = len(sev_scales)
    for i in range(1, num_sample):
        x_label = x_label + ', ' + str(sev_scales[i])
    return x_label


if args.test_mode_eval:
    GAN_4.eval()
    GAN_6.eval()
    GAN_8.eval()


# Reading Phenotype file
samples = []
pheno_file_lines = Data.read_pheno_file()
file_names = [x[0] for x in pheno_file_lines[1:]]
amd_sev = [x[9] for x in pheno_file_lines[1:]]
amd_sev_dict = {u: v for u, v in zip(file_names, amd_sev)}
del file_names, amd_sev

for idx in tqdm(range(args.num_prediction)):
    curr_train_seq, curr_val_seq, curr_test_seq = train_long_seq[idx], val_long_seq[idx], test_long_seq[idx]
    train_images, train_labels = load_seq_images(curr_train_seq, amd_sev_dict, args.im_resize_shape)
    val_images, val_labels = load_seq_images(curr_val_seq, amd_sev_dict, args.im_resize_shape)
    test_images, test_labels = load_seq_images(curr_test_seq, amd_sev_dict, args.im_resize_shape)

    # ##### Train
    if (train_images != []) and (train_labels != []):
        train_preds = make_prediction(train_images[0].unsqueeze(0), GAN_4, GAN_6, GAN_8)
        train_preds = torch.cat([train_images[0].unsqueeze(0), train_preds], dim=0)
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('off')
        axs[0].set_title("Baseline, Pred @ 4, Pred @ 6, Pred @ 8")
        axs[0].imshow(np.transpose(vutils.make_grid(train_preds, padding=2, normalize=True).cpu(), (1, 2, 0)))

        axs[1].axis('off')
        axs[1].set_title("Baseline, GT-4, GT-6, GT-8")
        axs[1].imshow(np.transpose(vutils.make_grid(train_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
        xlabel = sev_scale_xlabel(train_labels)
        fig.suptitle('Train image: {}'.format((curr_train_seq[0].split('/'))[-1]))
        fig.text(0.5, 0.11, xlabel, ha='center', fontsize=12)
        plt.savefig(os.path.join(args.stdout, ('Train_' + (curr_train_seq[0].split('/'))[-1])))
        plt.close()

    # ##### Val
    if (val_images != []) and (val_labels != []):
        val_preds = make_prediction(val_images[0].unsqueeze(0), GAN_4, GAN_6, GAN_8)
        val_preds = torch.cat([val_images[0].unsqueeze(0), val_preds], dim=0)
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('off')
        axs[0].set_title("Baseline, Pred @ 4, Pred @ 6, Pred @ 8")
        axs[0].imshow(np.transpose(vutils.make_grid(val_preds, padding=2, normalize=True).cpu(), (1, 2, 0)))

        axs[1].axis('off')
        axs[1].set_title("Baseline, GT-4, GT-6, GT-8")
        axs[1].imshow(np.transpose(vutils.make_grid(val_images, padding=2, normalize=True).cpu(), (1, 2, 0)))

        xlabel = sev_scale_xlabel(val_labels)
        fig.suptitle('Val image: {}'.format((curr_val_seq[0].split('/'))[-1]))
        fig.text(0.5, 0.11, xlabel, ha='center', fontsize=12)
        plt.savefig(os.path.join(args.stdout, ('Val_' + (curr_val_seq[0].split('/'))[-1])))
        plt.close()

    # ##### Test
    if (test_images != []) and (test_labels != []):
        test_preds = make_prediction(test_images[0].unsqueeze(0), GAN_4, GAN_6, GAN_8)
        test_preds = torch.cat([test_images[0].unsqueeze(0), test_preds], dim=0)
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('off')
        axs[0].set_title("Baseline, Pred @ 4, Pred @ 6, Pred @ 8")
        axs[0].imshow(np.transpose(vutils.make_grid(test_preds, padding=2, normalize=True).cpu(), (1, 2, 0)))

        axs[1].axis('off')
        axs[1].set_title("Baseline, GT-4, GT-6, GT-8")
        axs[1].imshow(np.transpose(vutils.make_grid(test_images, padding=2, normalize=True).cpu(), (1, 2, 0)))

        xlabel = sev_scale_xlabel(test_labels)
        fig.suptitle('Test image: {}'.format((curr_test_seq[0].split('/'))[-1]))
        fig.text(0.5, 0.11, xlabel, ha='center', fontsize=12)
        plt.savefig(os.path.join(args.stdout, ('Test_' + (curr_test_seq[0].split('/'))[-1])))
        plt.close()

logging.warning('GAN 4 epoch: {}'.format(epoch_4))
logging.warning('GAN 6 epoch: {}'.format(epoch_6))
logging.warning('GAN 8 epoch: {}'.format(epoch_8))
logging.warning('Dropout: {}'.format(not args.no_dropout))
