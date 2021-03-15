"""
Code for creating the saliency maps.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from skimage import io
from PIL import Image
from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import set_logging_settings, pair_visits
from Utils.utils import add_common_args
from Models.NextStepModels import Classifier
from collections import OrderedDict
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int,  default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='1', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--device_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=5, help='loads 2 samples in train/val/test datasets.')
parser.add_argument('--use_next', default=True, help='Whether using the second elements in pairs or the first ones.')
parser.add_argument('--num_save_image', default=50)
parser.add_argument('--channel_ratios', default=(10, 10, 10))
parser.add_argument('--saliency_th', default='median')
parser.add_argument('--remove_negatives', default=True)
parser.add_argument('--saliency_ratio', default=5.)

# Data Parameters
parser.add_argument('--balanced', default=True, help='Whether to discard samples to balance the dataset or not.')
parser.add_argument('--binary', default=False, help='If true, use both current and next label. Else, only next.')
parser.add_argument('--visit_gap', default=4, help='Time duration (in units of x6 months) between visit pairs.')
parser.add_argument('--late_AMD_threshold', default=10, help='9/10. The threshold score for late AMD')
parser.add_argument('--min_visit_no', default=0, help='Minimum possible visit number for a subject.')
parser.add_argument('--max_visit_no', default=26, help='Maximum possible visit number for a subject.')
parser.add_argument('--min_participant_pairs', default=1, help='min # of pairs to use for each eye in each participant')
parser.add_argument('--ns_data_split_ratio', default=[0.9, 0.05, 0.05])
parser.add_argument('--transform', default=['resize'])
parser.add_argument('--im_resize_shape', default=224)

# Classifier
parser.add_argument('--pretrained', default=True)
parser.add_argument('--classifier_net', default='resnet-18')
parser.add_argument('--num_class', default=3)

parser = add_common_args(parser)
args = parser.parse_args()

# ############################# Logging & Fixing Seed #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if int(args.gpus) >= 0:
    torch.cuda.manual_seed_all(args.random_seed)

if args.binary:
    args.num_class = 2

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])

# ############################# Loading Data #############################
args.visits = pair_visits(args.min_visit_no, args.max_visit_no, args.visit_gap)
Data = AMDDataAREDS(args)
Data.prepare_data()
Data.setup()
np.random.seed(args.random_seed)

if args.binary:
    Data.convert_to_binary()

partitions = ['whole', 'train', 'val', 'test']
for p in partitions:
    logging.warning('************************')
    logging.warning('Histogram of ' + p + ' partition:')
    Data.data_histogram(partition=p, log=True)
    logging.warning('************************')

# ############################# Defining Model and Loss Function #############################
classifier = Classifier(args)
if args.device_ids:
    classifier = nn.DataParallel(classifier, device_ids=args.device_ids)
classifier.to(args.device)

# ############################# Loading Checkpoint #############################
checkpoint_class_dir, checkpoint_file = '', ''
if args.visit_gap == 4:
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_4'
    checkpoint_file = 'class_chkpnt_iter_232.pth'
elif args.visit_gap == 6:
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_6'
    checkpoint_file = 'class_chkpnt_iter_1254.pth'
elif args.visit_gap == 8:
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_8'
    checkpoint_file = 'class_chkpnt_iter_146.pth'
else:
    raise NotImplementedError

file_name_class = os.path.join(checkpoint_class_dir, checkpoint_file)
logging.warning('Loading Checkpoint from: {}'.format(file_name_class))
checkpoint_class = torch.load(file_name_class, map_location=args.device)
for u, v in checkpoint_class.items():
    if (u == 'model') or (u == 'classifier'):
        continue
    logging.warning('{}: \t {:.4f}'.format(u, v))

try:
    checkpoint_model = checkpoint_class['model']
except KeyError:
    checkpoint_model = checkpoint_class['classifier']


if 'module' == (list(checkpoint_model.keys())[0])[:len('module')]:  # Trained in DataParallel mode.
    cls_state_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        name = k[7:]  # remove `module.`
        cls_state_dict[name] = v
    checkpoint_model = cls_state_dict

classifier.load_state_dict(checkpoint_model)
classifier.to(args.device)
classifier.eval()
# ############################# Creating Saliency Maps #############################


def load_image(im_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    im = io.imread(im_dir)
    image = Image.fromarray(im)
    sample_image = transform(image)
    return sample_image


def prepare_image_list(datamodule, partition):
    if partition == 'train':
        dataset = datamodule.train_set
    elif partition == 'val':
        dataset = datamodule.val_set
    elif partition == 'test':
        dataset = datamodule.test_set
    else:
        raise ValueError
    samples_pairs = []
    for sample_dict in dataset.samples_dict:
        for pair in sample_dict['LE']:
            samples_pairs.append(pair)
        for pair in sample_dict['RE']:
            samples_pairs.append(pair)
    random.shuffle(samples_pairs)
    return samples_pairs


def tensor_to_image(im_tensor):
    im_np = im_tensor.cpu().numpy()
    image = Image.fromarray(im_np, 'RGB')
    return image


def find_amd_sev_scale(im_path, amd_sev_dict):
    img_name = ((im_path.split('/'))[-1])[:-4]
    img_name_jpg = img_name + '.jpg'
    img_name_dng = img_name + '.dng'
    amd_sev_keys = list(amd_sev_dict.keys())
    if img_name_jpg in amd_sev_keys:
        return amd_sev_dict[img_name_jpg]
    elif img_name_dng in amd_sev_keys:
        return amd_sev_dict[img_name_dng]
    else:
        return []


def load_pair(pair_info, amd_sev_dict):
    pair_images, pair_sev_scores = [], []
    pair_label = pair_info[1]
    for img_idx in range(2):
        sev_score = find_amd_sev_scale(pair_info[0][img_idx], amd_sev_dict)
        if not sev_score:
            return [], [], []
        else:
            pair_sev_scores.append(sev_score)
        pair_images.append(load_image(pair_info[0][img_idx]))
    pair_images = torch.stack(pair_images, 0).squeeze()
    return pair_images, pair_label, pair_sev_scores


def make_prediction_and_figure(pair, model, amd_sev_dict, saliency_th='median', remove_negatives=False,
                               channel_ratios=(10, 10, 10), partition='Train'):
    """
    :param pair_images: A tuple with the form (current image name, next time point image name, label).
    :param model: The classifier model.
    :param amd_sev_dict: A dictionary containing {image name: AMD Sev Score} values for all images.
    :param saliency_th: Whether to use median/average of non-zero gradients for thresholding before drawing saliency
     maps.
    :return:
    """
    pair_images, pair_label, pair_sev_scores = load_pair(pair, amd_sev_dict)
    if (pair_images == []) or (pair_label == []) or (pair_sev_scores == []):
        return
    x = pair_images[0].unsqueeze(0)
    x.requires_grad_()
    model.eval()
    preds = model(x)
    preds_max_idx = preds.argmax()
    preds_max = preds[0, preds_max_idx]
    preds_max.backward()
    g = x.grad.data
    if remove_negatives:
        g[g < 0] = 0.
    saliency, _ = torch.max(g.abs(), dim=1)
    saliency_positive = saliency[saliency > 0]
    th = torch.max(saliency_positive) / args.saliency_ratio
    saliency[saliency <= th] = 0

    first_img = Image.fromarray((x.detach().squeeze().permute(1, 2, 0).numpy() * 255).astype('uint8'), 'RGB')
    sal_img = Image.fromarray((saliency.squeeze().numpy() * 255).astype('uint8'), 'L')
    sal_img = sal_img.convert('RGB')
    r, g, b = sal_img.split()
    r_red, r_blue, r_green = channel_ratios[0], channel_ratios[1], channel_ratios[2]
    r = r.point(lambda i: i * r_red)
    g = g.point(lambda i: i * r_green)
    b = b.point(lambda i: i * r_blue)
    sal_img = Image.merge('RGB', (r, g, b))
    blend = Image.blend(first_img, sal_img, 0.5)
    # blend_tensor = torch.from_numpy(np.array(blend))
    blend_tensor = transforms.ToTensor()(blend)
    all_images = torch.cat([pair_images[0].unsqueeze(0), blend_tensor.unsqueeze(0), pair_images[1].unsqueeze(0)], dim=0)
    fig, ax = plt.subplots(1, 1)
    plt.xticks([])
    plt.yticks([])
    ax.set_title("First Time Point, Saliency Map, Next Time Point")
    ax.imshow(np.transpose(vutils.make_grid(all_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    ax.set_xlabel('Severity Scales: {}, {}, Pair Label: {}, Prediction: {}'.format(pair_sev_scores[0], pair_sev_scores[1], pair_label, preds_max_idx))
    fig.suptitle('{} image: {}'.format(partition, (pair[0][0].split('/'))[-1]))
    plt.savefig(os.path.join(args.stdout, (partition + (pair[0][0].split('/'))[-1])))
    plt.close()

# Reading Phenotype file
samples = []
pheno_file_lines = Data.read_pheno_file()
file_names = [x[0] for x in pheno_file_lines[1:]]
amd_sev = [x[9] for x in pheno_file_lines[1:]]
amd_sev_dict = {u: v for u, v in zip(file_names, amd_sev)}
del file_names, amd_sev

train_pairs, val_pairs = prepare_image_list(Data, 'train'), prepare_image_list(Data, 'val')
test_pairs = prepare_image_list(Data, 'test')

for i in range(args.num_save_image):
    train_pair, val_pair, test_pair = train_pairs[i], val_pairs[i], test_pairs[i]
    make_prediction_and_figure(train_pair, classifier, amd_sev_dict, args.saliency_th, args.remove_negatives,
                               args.channel_ratios, 'Train')
    make_prediction_and_figure(val_pair, classifier, amd_sev_dict, args.saliency_th, args.remove_negatives,
                               args.channel_ratios, 'Val')
    make_prediction_and_figure(test_pair, classifier, amd_sev_dict, args.saliency_th, args.remove_negatives,
                               args.channel_ratios, 'Test')
