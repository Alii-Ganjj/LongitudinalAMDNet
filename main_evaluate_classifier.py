"""
Code for calculating Confidence Interval for checkpoints of the trained classifiers (main_classifier.py file) for
2, 3, and 4 years gap. It is used to calculate the values in the Table 2 and some of the ones in the Table 3 of the
paper.
Main Parameters:

- use_next: A flag which determines that the input to the classifier is the first image for the first time point in a
pair or the second one. I.e., if we denote a pair with "((I1, I2), label_pair)", use_next=True will result in using I1
as the input of the model and I2 vice versa.

- binary: A flag to determine the labeling scheme. If we denote a pair with "((I1, I2), label_pair)", binary=False will
result in the default 3 possible classes described in the paper {(not adv, not adv), (not adv, adv), (adv, adv)}, but
if binary=True, then the label for each pair will be determined based on the label of I2 only.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np
import torch.nn as nn

from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import set_logging_settings, pair_visits
from Utils.utils import add_common_args
from Models.NextStepModels import Classifier
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, confusion_matrix
from numpy import mean, median, percentile


parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int,  default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpus', default='1', help='gpu:i, i in [0, 1, 2, 3]')
parser.add_argument('--device_ids', default=[])
parser.add_argument('--debug', default=False, help='loads less training data to make debug faster.')
parser.add_argument('--debug_len', default=5, help='loads 2 samples in train/val/test datasets.')
parser.add_argument('--use_next', default=False, help='Whether using the second elements in pairs or the first ones.')

# Confidence Interval Calculation Parameters
parser.add_argument('--num_bootstrap', default=2000)
parser.add_argument('--alpha', default=5., help='confidence interval = (100 - alpha)')

# Data Parameters
parser.add_argument('--balanced', default=True, help='Whether to discard samples to balance the dataset or not.')
parser.add_argument('--binary', default=False, help='If False, use both current and next label. Else, only next.')
parser.add_argument('--visit_gap', default=4, help='Time duration (in units of x6 months) between visit pairs.')
parser.add_argument('--late_AMD_threshold', default=10, help='9/10. The threshold score for late AMD')
parser.add_argument('--min_visit_no', default=0, help='Minimum possible visit number for a subject.')
parser.add_argument('--max_visit_no', default=26, help='Maximum possible visit number for a subject.')
parser.add_argument('--min_participant_pairs', default=1, help='min # of pairs to use for each eye in each participant')
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

class_weights = Data.class_weights()
class_weights = class_weights.to(args.device)

# ############################# Defining Model and Loss Function #############################
classifier = Classifier(args)
if args.device_ids:
    classifier = nn.DataParallel(classifier, device_ids=args.device_ids)
classifier.to(args.device)
args.class_loss = nn.CrossEntropyLoss(weight=class_weights)

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


def perform_bootstrap(y_pred, y_true, num_bootstrap=2000, alpha=5.):
    num_data = y_pred.shape[0]
    indices = np.arange(num_data)
    num_test = 1
    bootstr_auc, bootstr_acc = [], []
    while num_test <= num_bootstrap:
        idx = np.random.choice(indices, num_data)
        if not ((np.unique(y_true[idx])).shape[0] == 3):
            continue
        num_test += 1
        auc = roc_auc_score(y_true[idx], y_pred[idx], multi_class='ovo')
        bootstr_auc.append(auc)
        y_pred_label = np.argmax(y_pred[idx], axis=1)
        acc = ((y_pred_label == y_true[idx]).sum()) / num_data
        bootstr_acc.append(acc)

    # Calculating CIs based on the bootstrap values.
    lower_p = alpha / 2.
    upper_p = 100 - alpha + alpha / 2
    # AUC
    lower_auc = max(0., percentile(bootstr_auc, lower_p))
    upper_auc = min(1., percentile(bootstr_auc, upper_p))
    med_auc = median(bootstr_auc)
    mean_auc = mean(bootstr_auc)
    # Acc
    lower_acc = max(0., percentile(bootstr_acc, lower_p))
    upper_acc = min(1., percentile(bootstr_acc, upper_p))
    med_acc = median(bootstr_acc)
    mean_acc = mean(bootstr_acc)
    return {'CI_auc': (lower_auc, upper_auc), 'CI_acc': (lower_acc, upper_acc),
            'median_auc': med_auc, 'mean_auc': mean_auc, 'median_acc': med_acc, 'mean_acc': mean_acc}


def test_next_step_classifier(model, test_dataloader, args, calc_CI=False, use_next=False):
    """
    :param model: The checkpoint model to evaluate.
    :param calc_CI: Boolean flag determining whether to calculate confidence interval or not.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            if not use_next:
                x, y = data[0][0].to(args.device), data[1].to(args.device)
            else:
                x, y = data[0][1].to(args.device), data[1].to(args.device)
            y_pred = model(x)
            y_preds.append(y_pred)
            y_true.append(y)
            loss = args.class_loss(y_pred, y)
            total_loss += loss
            _, label_pred = torch.max(y_pred, dim=1)
            total += y.shape[0]
            correct += (label_pred == y).sum().item()
    total_loss = total_loss / total
    accuracy = correct / total
    y_preds = torch.softmax(torch.stack(y_preds, dim=0).squeeze(), dim=1)
    y_true = torch.stack(y_true).squeeze()
    if not args.debug:
        if args.binary:
            auc = roc_auc_score(y_true.cpu().numpy(), (y_preds[:, 1].squeeze()).cpu().numpy())
        else:
            auc = roc_auc_score(y_true.cpu().numpy(), y_preds.cpu().numpy(), multi_class='ovo')
    else:
        auc = 0.

    if calc_CI:
        conf_intervals = perform_bootstrap(y_preds.cpu().numpy(), y_true.cpu().numpy(),
                                           num_bootstrap=args.num_bootstrap, alpha=args.alpha)
        return {'loss': total_loss, 'acc': accuracy, 'auc': auc, 'CI_auc': conf_intervals['CI_auc'],
                'CI_acc': conf_intervals['CI_acc'], 'median_auc': conf_intervals['median_auc'],
                'mean_auc': conf_intervals['mean_auc'], 'median_acc': conf_intervals['median_acc'],
                'mean_acc': conf_intervals['mean_acc']}, y_preds, y_true

    else:
        return {'loss': total_loss, 'acc': accuracy, 'auc': auc}, y_preds, y_true


val_dataloader, test_dataloader = Data.val_dataloader(), Data.test_dataloader()
eval_val, pred_val, y_val = test_next_step_classifier(classifier, val_dataloader, args, calc_CI=True)
eval_test, pred_test, y_test = test_next_step_classifier(classifier, test_dataloader, args, calc_CI=True)

logging.warning('Best Checkpoint Results on Validation set:')
for k, v in eval_val.items():
    if not ((k == 'CI_acc') or (k == 'CI_auc')):
        logging.warning('{}: {:.4f}'.format(k, v))
    else:
        logging.warning('{}: ({:.4f}, {:.4f})'.format(k, v[0], v[1]))

logging.warning('Best Checkpoint Results on Test set:')
for k, v in eval_test.items():
    if not ((k == 'CI_acc') or (k == 'CI_auc')):
        logging.warning('{}: {:.4f}'.format(k, v))
    else:
        logging.warning('{}: ({:.4f}, {:.4f})'.format(k, v[0], v[1]))

# Confusion Matrix
_, label_pred_val = torch.max(pred_val, 1)
val_conf_matrix = confusion_matrix(y_val.cpu().numpy(), label_pred_val.cpu().numpy())
logging.warning('Validation Confusion Matrix: ')
logging.warning(val_conf_matrix)

_, label_pred_test = torch.max(pred_test, 1)
te_conf_matrix = confusion_matrix(y_test.cpu().numpy(), label_pred_test.cpu().numpy())
logging.warning('Test Confusion Matrix: ')
logging.warning(te_conf_matrix)

if not args.use_next:
    eval_val, pred_val, y_val = test_next_step_classifier(classifier, val_dataloader, args, calc_CI=True)
    eval_test, pred_test, y_test = test_next_step_classifier(classifier, test_dataloader, args, calc_CI=True)
    logging.warning('Best Checkpoint Results on Validation set:')
    for k, v in eval_val.items():
        if not ((k == 'CI_acc') or (k == 'CI_auc')):
            logging.warning('{}: {:.4f}'.format(k, v))
        else:
            logging.warning('{}: ({:.4f}, {:.4f})'.format(k, v[0], v[1]))
    logging.warning('Best Checkpoint Results on Test set:')
    for k, v in eval_test.items():
        if not ((k == 'CI_acc') or (k == 'CI_auc')):
            logging.warning('{}: {:.4f}'.format(k, v))
        else:
            logging.warning('{}: ({:.4f}, {:.4f})'.format(k, v[0], v[1]))
    # Confusion Matrix
    _, label_pred_val = torch.max(pred_val, 1)
    val_conf_matrix = confusion_matrix(y_val.cpu().numpy(), label_pred_val.cpu().numpy())
    logging.warning('Validation Confusion Matrix: ')
    logging.warning(val_conf_matrix)
    # ## Added 13 Jan
    label_pred_val_b = np.copy(label_pred_val.cpu().numpy())
    label_pred_val_b[label_pred_val_b != 0] = 1
    y_val_b = np.copy(y_val.cpu().numpy())
    y_val_b[y_val_b != 0] = 1
    val_b_conf_matrix = confusion_matrix(y_val_b, label_pred_val_b)
    logging.warning('Validation Binary Confusion Matrix: ')
    logging.warning(val_b_conf_matrix)
    acc_val = (val_b_conf_matrix[0, 0] + val_b_conf_matrix[1, 1]) / (val_b_conf_matrix[0, 0] + val_b_conf_matrix[0, 1] +
                                                                     val_b_conf_matrix[1, 0] + val_b_conf_matrix[1, 1])
    logging.warning('Acc Val: {:.4f}'.format(acc_val))

    _, label_pred_test = torch.max(pred_test, 1)
    te_conf_matrix = confusion_matrix(y_test.cpu().numpy(), label_pred_test.cpu().numpy())
    logging.warning('Test Confusion Matrix: ')
    logging.warning(te_conf_matrix)
    # ## Added 13 Jan
    label_pred_te_b = np.copy(label_pred_test.cpu().numpy())
    label_pred_te_b[label_pred_te_b != 0] = 1
    y_te_b = np.copy(y_test.cpu().numpy())
    y_te_b[y_te_b != 0] = 1
    test_b_conf_matrix = confusion_matrix(y_te_b, label_pred_te_b)
    logging.warning('Test Binary Confusion Matrix: ')
    logging.warning(test_b_conf_matrix)
    acc_test = (test_b_conf_matrix[0, 0] + test_b_conf_matrix[1, 1]) / (test_b_conf_matrix[0, 0] +
                                                                        test_b_conf_matrix[0, 1] +
                                                                        test_b_conf_matrix[1, 0] +
                                                                        test_b_conf_matrix[1, 1])
    logging.warning('Acc Test: {:.4f}'.format(acc_test))

else:
    _, pred_val, y_val = test_next_step_classifier(classifier, val_dataloader, args, use_next=True)
    _, pred_test, y_test = test_next_step_classifier(classifier, test_dataloader, args, use_next=True)
    _, label_pred_val = torch.max(pred_val, 1)
    label_pred_val_b = np.copy(label_pred_val.cpu().numpy())
    label_pred_val_b[label_pred_val_b <= 1] = 0
    label_pred_val_b[label_pred_val_b == 2] = 1
    y_val_b = np.copy(y_val.cpu().numpy())
    y_val_b[y_val_b <= 1] = 0
    y_val_b[y_val_b == 2] = 1
    val_b_conf_matrix = confusion_matrix(y_val_b, label_pred_val_b)
    logging.warning('Validation Binary Confusion Matrix: ')
    logging.warning(val_b_conf_matrix)
    acc_val = (val_b_conf_matrix[0, 0] + val_b_conf_matrix[1, 1]) / (val_b_conf_matrix[0, 0] +
                                                                     val_b_conf_matrix[0, 1] +
                                                                     val_b_conf_matrix[1, 0] +
                                                                     val_b_conf_matrix[1, 1])
    logging.warning('Acc Val: {:.4f}'.format(acc_val))

    _, label_pred_test = torch.max(pred_test, 1)
    label_pred_te_b = np.copy(label_pred_test.cpu().numpy())
    label_pred_te_b[label_pred_te_b <= 1] = 0
    label_pred_te_b[label_pred_te_b == 2] = 1
    y_te_b = np.copy(y_test.cpu().numpy())
    y_te_b[y_te_b <= 1] = 0
    y_te_b[y_te_b == 2] = 1
    test_b_conf_matrix = confusion_matrix(y_te_b, label_pred_te_b)
    logging.warning('Test Binary Confusion Matrix: ')
    logging.warning(test_b_conf_matrix)
    acc_test = (test_b_conf_matrix[0, 0] + test_b_conf_matrix[1, 1]) / (test_b_conf_matrix[0, 0] +
                                                                        test_b_conf_matrix[0, 1] +
                                                                        test_b_conf_matrix[1, 0] +
                                                                        test_b_conf_matrix[1, 1])
    logging.warning('Acc Test: {:.4f}'.format(acc_test))
