import os
import sys
import logging
import time
import socket
import torch

from argparse import ArgumentParser
from Utils.allenNLP_tee_logger import TeeLogger


def prepare_global_logging(serialization_dir: str, file_friendly_logging: bool) -> logging.FileHandler:
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serialization_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carriage returns
        (used to update progress bars on a single terminal line). This
        option is typically only used if you are running in an environment
        without a terminal.

    Returns
    -------
    ``logging.FileHandler``
        A logging file handler that can later be closed and removed from the global logger.
    """

    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    # Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, "stdout.log")
    sys.stdout = TeeLogger(std_out_file, # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

    return stdout_handler


def logging_args(args):
    for arg in vars(args):
        logging.warning(str(arg) + ': ' + str(getattr(args, arg)))


def set_logging_settings(args, main_file):
    # args.data_dir = './Datasets/AREDS'
    # args.stdout = './results'
    # args.data_ukb_dir = './Datasets/AMDUKB'
    args.data_dir = '/home/alireza/Datasets/AMDImageGenetics'
    args.stdout = '/home/alireza/results'
    args.data_ukb_dir = '/home/alireza/Datasets/AMDImageGeneticCheck/AMDUKB'

    args.pheno_dir = os.path.join(args.data_dir, args.phenotype_file)
    args.pheno_ukb_dir = os.path.join(args.data_ukb_dir, args.phenotype_file_ukb)
    args.image_dict = os.path.join(args.data_dir, args.image_dict_file)
    args.image_dict_ukb = os.path.join(args.data_ukb_dir, args.image_dict_file_ukb)

    args.image_dir = {'original_2014': os.path.join(args.data_dir, 'img_2014'),
                      'original_2010': os.path.join(args.data_dir, 'img_2010'),
                      '2010': os.path.join(args.data_dir, 'img_2010_resized_deepseenet'),
                      '2014': os.path.join(args.data_dir, 'img_2014_resized_deepseenet'),
                      'ukb_original_RE': os.path.join(args.data_ukb_dir, 'RE'),
                      'ukb_original_LE': os.path.join(args.data_ukb_dir, 'LE'),
                      'ukb_RE': os.path.join(args.data_ukb_dir, 'RE_resized_deepseenet'),
                      'ukb_LE': os.path.join(args.data_ukb_dir, 'LE_resized_deepseenet')}

    args.device = torch.device(f"cuda:{args.gpus}" if (torch.cuda.is_available()) else "cpu")
    type = args.device.type
    index = str(args.device.index) if (isinstance(args.device.index, int)) else ''
    args.stdout = os.path.join(args.stdout, os.path.basename(os.getcwd()), main_file,
                               time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") + socket.gethostname() + '_' +
                               type + index)
    checkpoint_folder = os.path.join(args.data_dir, 'checkpoints')
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    checkpoint_dir = os.path.join(checkpoint_folder, main_file)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    args.checkpoint_dir = os.path.join(checkpoint_dir, time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") +
                                       socket.gethostname() + '_' + type + index)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=args.logging_level)
    prepare_global_logging(serialization_dir=args.stdout, file_friendly_logging=False)
    if args.verbose > 0:
        logging_args(args)
    return args


def pair_visits(min_visit_no, max_visit_no, visit_gap):
    assert (min_visit_no + visit_gap <= max_visit_no)
    possible_pairs = []
    for visit_no in range(min_visit_no, max_visit_no - visit_gap + 1):
        pair = ['{0:02d}'.format(visit_no), '{0:02d}'.format(visit_no + visit_gap)]
        possible_pairs.append(pair)
    return possible_pairs


def add_common_args(parent_parser):
    # Files Information.
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--logging_level', default=logging.INFO, help='Options: logging.DEBUG, logging.INFO')
    parser.add_argument('--phenotype_file', type=str, default='image_pheno_master.txt')
    parser.add_argument('--phenotype_file_ukb', type=str, default='UKB_master_pheno.csv')
    parser.add_argument('--pheno_file_lines_info_dict',
                        default={'AMDSEV': 9, 'DRSZWI': 10, 'ELICAT': 11, 'DRUSF2': 12, 'NDRUF2': 13, 'DRSOFT': 26})
    parser.add_argument('--image_dict_file', default='image_dic_2.txt',
                        help='A dictionary containing hierarchical info of data.')
    parser.add_argument('--image_dict_file_ukb', default='UKB_image_dic.txt')

    # Data Parameters
    parser.add_argument('--im_size', default=224, help='size for resizing the images.')
    parser.add_argument('--seed_data', default=1, help='random seed used for partitioning the data.')
    parser.add_argument('--data_split_ratio', default=[0.9, 0.05, 0.05])
    return parser
