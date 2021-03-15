"""
This script loads a checkpoint from Pix2Pix and a checkpoint of a classifier. It inputs the predicted 'next step' images
from the generator into the discriminator and compares the results with the ground truth labels.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import set_logging_settings, pair_visits, add_common_args
from Models.NextStepModels import Pix2PixModel, Classifier
from collections import OrderedDict
from sklearn.metrics import confusion_matrix


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
parser.add_argument('--epoch', default=175, help="checkpoint's epoch number.")
parser.add_argument('--binary_classifier', default=False)

# Data Parameters
parser.add_argument('--visit_gap', default=6, help='Time duration (in units of x6 months) between visit pairs.')
parser.add_argument('--late_AMD_threshold', default=10, help='9/10. The threshold score for late AMD')
parser.add_argument('--min_visit_no', default=0, help='Minimum possible visit number for a subject.')
parser.add_argument('--max_visit_no', default=26, help='Maximum possible visit number for a subject.')
parser.add_argument('--min_participant_pairs', default=1, help='min # of pairs to use for each eye in each participant')
parser.add_argument('--transform', default=['resize'])
parser.add_argument('--im_resize_shape', default=256)
parser.add_argument('--classifier_input_size', default=224)
parser.add_argument('--binary', default=True)
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
parser.add_argument('--no_dropout', default=False, help='no dropout for the generator')

# Classifier
parser.add_argument('--pretrained', default=True)
parser.add_argument('--classifier_net', default='resnet-18')
parser.add_argument('--num_class', default=3)
parser.add_argument('--classifier_test_eval_mode', default=True)


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
Data.convert_to_binary()
np.random.seed(args.random_seed)

partitions = ['whole', 'train', 'val', 'test']
for p in partitions:
    logging.warning('************************')
    logging.warning('Histogram of ' + p + 'partition:')
    Data.data_histogram(partition=p, log=True)
    logging.warning('************************')


val_dataloader, test_dataloader = Data.val_dataloader(), Data.test_dataloader()

# ############################# Loading Checkpoints #############################
if args.binary_classifier:
    checkpoint_class_dir = './Datasets/checkpoints/binary_classifier'
    checkpoint_file = 'class_chkpnt_iter_352.pth'
    args.num_class = 2
if args.visit_gap == 4:
    args.checkpoint_dir = './Datasets/checkpoints/GAN_4'
    args.epoch = 35
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_4'
    checkpoint_file = 'class_chkpnt_iter_232.pth'
elif args.visit_gap == 6:
    args.checkpoint_dir = './Datasets/checkpoints/GAN_6'
    args.epoch = 40
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_6'
    checkpoint_file = 'class_chkpnt_iter_1254.pth'
elif args.visit_gap == 8:
    args.checkpoint_dir = './Datasets/checkpoints/GAN_8'
    args.epoch = 45
    checkpoint_class_dir = './Datasets/checkpoints/Classifier_8'
    checkpoint_file = 'class_chkpnt_iter_146.pth'
else:
    raise NotImplementedError

model_GAN = Pix2PixModel(args, isTrain=False).to(args.device)
model_GAN.setup(args)

checkpoint_class_dir = os.path.join(checkpoint_class_dir, checkpoint_file)
checkpoint_class = torch.load(checkpoint_class_dir, map_location=args.device)

classifier = Classifier(args).to(args.device)

try:
    checkpoint_model = checkpoint_class['model']
except:
    checkpoint_model = checkpoint_class['classifier']

# If the model is trained in distributed
if 'module' == (list(checkpoint_model.keys())[0])[:len('module')]:
    cls_state_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        name = k[7:]  # remove `module.`
        cls_state_dict[name] = v
    checkpoint_model = cls_state_dict

classifier.load_state_dict(checkpoint_model)
logging.warning('Loading checkpoint from: {} \n iter: {} \n val_loss {:.4f} \n val_acc {:.4f} \n val_AUC {:.4f} \n'
                ' test_loss {:.4f} \n test_acc {:.4f} \n test_AUC {:.4f}'.
                format(checkpoint_class_dir, checkpoint_class['iter'], checkpoint_class['val_loss'],
                       checkpoint_class['val_acc'], checkpoint_class['val_auc'], checkpoint_class['test_loss'],
                       checkpoint_class['test_acc'], checkpoint_class['test_auc']))


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


def resize_predicted_images(predicted_images, target_size):
    resized_images = []
    # transformer = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
    for img in predicted_images:
        # pil_img = transforms.ToPILImage()(img)
        # new_img = transformer(pil_img)
        new_img = F.interpolate(img.unsqueeze(0), (target_size, target_size))
        resized_images.append(new_img)
    resized_images = torch.stack(resized_images).squeeze()
    return resized_images


def evaluate_classifier(model, predicted_images, y, opt):
    model.train()
    if opt.classifier_test_eval_mode:
        model.eval()
    y_preds = []
    for i in range(predicted_images.shape[0]):
        y_pred = model(predicted_images[i].unsqueeze(0))
        y_preds.append(y_pred)
    y_preds = torch.stack(y_preds, 0).squeeze()
    _, label_pred = torch.max(y_preds, 1)
    label_pred[label_pred <= 1] = 0  # predicted label for predicted image is {0, 1} -> our prediction is not Adv AMD
    label_pred[label_pred == 2] = 1
    correct = (label_pred == y).sum().item()
    total = label_pred.shape[0]
    conf_mat = confusion_matrix(y.cpu().numpy(), label_pred.cpu().numpy())
    return conf_mat, correct / total


model_GAN.train()
classifier.train()
if args.test_mode_eval:
    model_GAN.eval()
if args.classifier_test_eval_mode:
    classifier.eval()

y_pred_val, y_val = [], []
for i, sample in enumerate(val_dataloader, 0):
    x1, x2, y = sample[0][0].to(args.device), sample[0][1].to(args.device), sample[1].to(args.device)
    y_val.append(y)
    model_GAN.set_input((x1, x2))
    model_GAN.test()
    predicted_image = model_GAN.fake_B

    # ## resizing predicted image
    predicted_image = F.interpolate(predicted_image, (args.classifier_input_size, args.classifier_input_size))

    # classification
    with torch.no_grad():
        y_pred = classifier(predicted_image)
    del x1, x2, predicted_image
    y_pred_val.append(y_pred)

y_pred_test, y_test = [], []
for i, sample in enumerate(test_dataloader, 0):
    x1, x2, y = sample[0][0].to(args.device), sample[0][1].to(args.device), sample[1].to(args.device)
    y_test.append(y)
    model_GAN.set_input((x1, x2))
    model_GAN.test()
    predicted_image = model_GAN.fake_B

    # ## resizing predicted image
    predicted_image = F.interpolate(predicted_image, (args.classifier_input_size, args.classifier_input_size))

    # classification
    with torch.no_grad():
        y_pred = classifier(predicted_image)
    del x1, x2, predicted_image
    y_pred_test.append(y_pred)

if not args.binary_classifier:
    y_val, y_test = torch.stack(y_val, 0).squeeze().to(args.device), torch.stack(y_test, 0).squeeze().to(args.device)
    y_pred_val, y_pred_test = torch.stack(y_pred_val, 0).squeeze().to(args.device), torch.stack(y_pred_test, 0).squeeze().to(args.device)
    _, label_pred_val = torch.max(y_pred_val, 1)
    _, label_pred_test = torch.max(y_pred_test, 1)
    label_pred_val[label_pred_val <= 1] = 0
    label_pred_val[label_pred_val == 2] = 1
    label_pred_test[label_pred_test <= 1] = 0
    label_pred_test[label_pred_test == 2] = 1

    correct_val, correct_test = (label_pred_val == y_val).sum().item(), (label_pred_test == y_test).sum().item()
    acc_val = correct_val / y_val.shape[0]
    conf_mat_val = confusion_matrix(y_val.cpu().numpy(), label_pred_val.cpu().numpy())

    logging.warning('Val acc: {:.4f}'.format(acc_val))
    logging.warning('Confusion Matrix Val: ')
    logging.warning(conf_mat_val)

    acc_test = correct_test / y_test.shape[0]
    conf_mat_test = confusion_matrix(y_test.cpu().numpy(), label_pred_test.cpu().numpy())

    logging.warning('Test acc: {:.4f}'.format(acc_test))
    logging.warning('Confusion Matrix: ')
    logging.warning(conf_mat_test)

else:
    y_val, y_test = torch.stack(y_val, 0).squeeze().to(args.device), torch.stack(y_test, 0).squeeze().to(args.device)
    y_pred_val, y_pred_test = torch.stack(y_pred_val, 0).squeeze().to(args.device), \
                              torch.stack(y_pred_test, 0).squeeze().to(args.device)
    _, label_pred_val = torch.max(y_pred_val, 1)
    _, label_pred_test = torch.max(y_pred_test, 1)

    correct_val, correct_test = (label_pred_val == y_val).sum().item(), (label_pred_test == y_test).sum().item()
    acc_val = correct_val / y_val.shape[0]
    conf_mat_val = confusion_matrix(y_val.cpu().numpy(), label_pred_val.cpu().numpy())

    logging.warning('Val acc: {:.4f}'.format(acc_val))
    logging.warning('Confusion Matrix Val: ')
    logging.warning(conf_mat_val)

    acc_test = correct_test / y_test.shape[0]
    conf_mat_test = confusion_matrix(y_test.cpu().numpy(), label_pred_test.cpu().numpy())

    logging.warning('Test acc: {:.4f}'.format(acc_test))
    logging.warning('Confusion Matrix: ')
    logging.warning(conf_mat_test)

logging.warning(args.epoch)
logging.warning(args.checkpoint_dir)
logging.warning(checkpoint_class_dir)
logging.warning(checkpoint_file)
