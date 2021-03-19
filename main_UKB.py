"""
Code for the experiment results for UK Bio-bank dataset.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np
import torch.nn as nn

from Utils.utils import set_logging_settings
from Utils.utils import add_common_args
from Models.NextStepModels import Classifier
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import skimage.io as io
import PIL.Image as Image

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='current_step')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='0', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--device_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=5, help='loads 2 samples in train/val/test datasets.')

# Data Parameters
parser.add_argument('--visit_gap', default=4, help='Time duration (in units of x6 months) between visit pairs.')

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

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])

# ############################# Defining Model and Loss Function #############################
classifier = Classifier(args)
if args.device_ids:
    classifier = nn.DataParallel(classifier, device_ids=args.device_ids)
classifier.to(args.device)

# ############################# Loading Checkpoint #############################
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


def load_image(im_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    im = io.imread(im_dir)
    image = Image.fromarray(im)
    sample_image = transform(image)
    return sample_image.unsqueeze(0)

classifier.eval()
y_pred = []
counter = 0
y_true = []

if args.experiment == 'current_step':
    # Predicting Current step

    Data_folder = args.data_ukb_dir
    label = {'AdvAMD': 1, 'Control': 0}
    for folder in ['AdvAMD', 'Control']:
        folder_label = label[folder]
        for eye in ['Left_Eye', 'Right_Eye']:
            directory = os.path.join(Data_folder, folder, eye)
            image_names = os.listdir(directory)
            image_names.sort()
            for name in image_names:
                if name[:2] == '._':
                    os.remove(os.path.join(directory, name))
                    continue
                counter += 1
                image = load_image(os.path.join(directory, name))
                image = image.to(args.device)
                y_true.append(torch.tensor(folder_label))
                pred = classifier(image)
                y_pred.append(pred)
                softmx = torch.softmax(pred, 1)
                _, label_prediction = torch.max(softmx, 1)
                if label_prediction <= 1:
                    label_prediction = 0
                else:
                    label_prediction = 1
                softmx = softmx.squeeze()
                logging.warning('#{}, {}, {}, {} \t Prediction: [{:.4f}, {:.4f}, {:.4f}], label: {}'.
                                format(counter, folder, eye, name, softmx[0], softmx[1], softmx[2], label_prediction))

    y_true = torch.stack(y_true, 0).squeeze().to(args.device)
    y_pred = torch.stack(y_pred, 0).squeeze().to(args.device)
    _, label_pred = torch.max(y_pred, dim=1)
    label_pred[label_pred <= 1] = 0
    label_pred[label_pred == 2] = 1
    correct = (label_pred == y_true).sum().item()
    acc = correct / y_pred.shape[0]
    conf_matrix = confusion_matrix(y_true.cpu().numpy(), label_pred.cpu().numpy())
    logging.warning(conf_matrix)
    logging.warning('Acc: {:.4f}'.format(acc))
