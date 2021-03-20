"""
This script loads original images from the AREDS and UK Bio-bank datasets and performs pre-processing steps described
in the paper on them.
"""
import os
import argparse
from Data.AMDDataAREDS import AMDDataAREDS
from Utils.utils import add_common_args, set_logging_settings

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--dataset', default='UKB', help='AREDS, UKB')
parser = add_common_args(parser)
args = parser.parse_args()
args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])

Data = AMDDataAREDS(args)

if args.dataset == 'AREDS':
    Data.resize_2014_images_deepseenet()
    Data.resize_2010_images_deepseenet()
elif args.dataset == 'UKB':
    Data.resize_ukb_RE_images_deepseenet()
    Data.resize_ukb_LE_images_deepseenet()
