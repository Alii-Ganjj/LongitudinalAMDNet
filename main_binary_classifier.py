"""
This script is used for training a binary classifier which predicts whether its input is in late-AMD stage or not.
The trained model is used in the GAN evaluation section of the paper
"""

import os
import logging
import argparse
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn

from training import train
from torch.utils.tensorboard import SummaryWriter
from Data.AMDDataCurrentStep import AMDDataCurrentStep
from Utils.utils import set_logging_settings, add_common_args
from Models.NextStepModels import Classifier
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int,  default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='1', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--device_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=5, help='loads 2 samples in train/val/test datasets.')
parser.add_argument('--num_epochs', default=20)
parser.add_argument('--num_test_epochs', default=1)
parser.add_argument('--num_test_iters', default=50)
parser.add_argument('--batch_size_train', default=128)

# Data Parameters
parser.add_argument('--balanced', default=False, help='Whether to discard samples to balance the dataset or not.')
parser.add_argument('--downsample_train', default=True,
                    help='If true, randomly down-samples the train dataset to lower the training time.')
parser.add_argument('--downsample_ratio', default=3)
parser.add_argument('--binary', default=False, help='If true, use both current and next label. Else, only next.')
parser.add_argument('--late_AMD_threshold', default=10, help='{9, 10}. The threshold score for late AMD')
parser.add_argument('--min_participant_pairs', default=1, help='min # of images to use for each eye in each subject')
parser.add_argument('--transform', default=['resize'])
parser.add_argument('--im_resize_shape', default=224)

# Classifier
parser.add_argument('--pretrained', default=True)
parser.add_argument('--classifier_net', default='resnet-18')
parser.add_argument('--num_class', default=2)

# Optimizer Parameters
parser.add_argument('--lr', default=3e-4)
parser.add_argument('--beta1', default=0.9)
parser.add_argument('--beta2', default=0.99)
parser.add_argument('--weight_decay', default=0.)

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
    args.batch_size_train = 2
    args.num_epochs = 1

np.random.seed(args.seed_data)
Data = AMDDataCurrentStep(args)
Data.prepare_data()
Data.setup()
if args.downsample_train:
    Data.train_set.downsample_pairs(ratio=args.downsample_ratio)
np.random.seed(args.random_seed)

partitions = ['whole', 'train', 'val', 'test']
for p in partitions:
    logging.warning('************************')
    logging.warning('Histogram of ' + p + ' partition:')
    Data.data_histogram(partition=p, log=True)
    logging.warning('************************')

class_weights = Data.class_weights()
class_weights = class_weights.to(args.device)

# ############################# Defining Model and Optimizer #############################
classifier = Classifier(args)
if args.device_ids:
    classifier = nn.DataParallel(classifier, device_ids=args.device_ids)
classifier.to(args.device)
args.optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                            weight_decay=args.weight_decay)
args.criterion = nn.CrossEntropyLoss(weight=class_weights)

# ############################# Training Model #############################
checkpoint_class = train(classifier, Data, args)
logging.warning('Best: iter: {} \t val_loss {:.4f} \t val_acc {:.4f} \t val_AUC {:.4f}'.
                format(checkpoint_class['iter'], checkpoint_class['val_loss'], checkpoint_class['val_acc'],
                       checkpoint_class['val_auc']))
logging.warning('test_loss {:.4f} \t test_acc {:.4f} \t test_AUC {:.4f}'.format(checkpoint_class['test_loss'],
                                                                                checkpoint_class['test_acc'],
                                                                                checkpoint_class['test_auc']))

checkpoint_class_name = 'class_chkpnt_iter_{}.pth'.format(checkpoint_class['iter'])
file_name_class = os.path.join(args.checkpoint_dir, checkpoint_class_name)
logging.warning('Saving Checkpoint: {}'.format(file_name_class))
torch.save(checkpoint_class, file_name_class)

logging.warning('Evaluating the best checkpoint (sanity check):')


def test_current_step_classifier(model, test_dataloader, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            x, y = data[0].to(args.device), data[1].to(args.device)
            y_pred = model(x)
            y_preds.append(y_pred)
            y_true.append(y)
            loss = args.criterion(y_pred, y)
            total_loss += loss
            _, label_pred = torch.max(y_pred, dim=1)
            total += y.shape[0]
            correct += (label_pred == y).sum().item()
    total_loss = total_loss / total
    accuracy = correct / total
    y_preds = torch.softmax(torch.stack(y_preds, dim=0).squeeze(), dim=1)
    y_true = torch.stack(y_true).squeeze()
    if not args.debug:
        if args.num_class == 2:
            auc = roc_auc_score(y_true.cpu().numpy(), (y_preds[:, 1].squeeze()).cpu().numpy())
        else:
            auc = roc_auc_score(y_true.cpu().numpy(), y_preds.cpu().numpy(), multi_class='ovo')
    else:
        auc = 0.
    return {'loss': total_loss, 'acc': accuracy, 'auc': auc}, y_preds, y_true


checkpoint_model = checkpoint_class['model']
if 'module' == (list(checkpoint_model.keys())[0])[:len('module')]:
    cls_state_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        name = k[7:]  # remove `module.`
        cls_state_dict[name] = v
    checkpoint_model = cls_state_dict

classifier.load_state_dict(checkpoint_model)
classifier.to(args.device)
val_dataloader, test_dataloader = Data.val_dataloader(), Data.test_dataloader()
eval_val, pred_val, y_val = test_current_step_classifier(classifier, val_dataloader, args)
eval_test, pred_test, y_test = test_current_step_classifier(classifier, test_dataloader, args)

logging.warning('Best Checkpoint Results on Validation set:')
for k, v in eval_val.items():
    logging.warning('{}: {:.4f}'.format(k, v))

logging.warning('Best Checkpoint Results on Test set:')
for k, v in eval_test.items():
    logging.warning('{}: {:.4f}'.format(k, v))

# Confusion Matrix
_, label_pred_val = torch.max(pred_val, 1)
val_conf_matrix = confusion_matrix(y_val.cpu().numpy(), label_pred_val.cpu().numpy())
logging.warning('Validation Confusion Matrix: ')
logging.warning(val_conf_matrix)

_, label_pred_test = torch.max(pred_test, 1)
te_conf_matrix = confusion_matrix(y_test.cpu().numpy(), label_pred_test.cpu().numpy())
logging.warning('Test Confusion Matrix: ')
logging.warning(te_conf_matrix)
