import os
import time
import torch.utils.data
import logging
import json
import pickle

import numpy as np
import torchvision.transforms as transforms
import pytorch_lightning as pl

from keras.preprocessing import image
from skimage import io
from torch.utils.data import DataLoader
from PIL import Image


class AMDDatasetAREDS(torch.utils.data.Dataset):
    def __init__(self, samples_dict, args, transform=None):
        self.args = args
        self.samples_dict = samples_dict
        self.pairs = []
        self.num_left_seq, self.num_right_seq = 0, 0
        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        pair_names = self.pairs[item][0]
        pair = []
        for img_name in pair_names:
            image = io.imread(img_name)
            image = Image.fromarray(image)
            sample_image = self.transforms(image)
            pair.append(sample_image)
        pair = (pair[0], pair[1])
        label = self.pairs[item][1]
        return (pair, label)

    def __len__(self):
        if self.args.debug:
            return self.args.debug_len
        else:
            return len(self.pairs)

    def prepare_pairs(self, balanced=False):
        if balanced:
            new_samples_dict = []
            for sample_dict in self.samples_dict:
                new_sample_dict = {'ID': sample_dict['ID'], 'LE': None, 'RE': None}
                if sample_dict['LE']:
                    new_sample_dict['LE'] = self.select_pairs(sample_dict['LE'])
                else:
                    new_sample_dict['LE'] = []
                if sample_dict['RE']:
                    new_sample_dict['RE'] = self.select_pairs(sample_dict['RE'])
                else:
                    new_sample_dict['RE'] = []
                new_samples_dict.append(new_sample_dict)
            self.samples_dict = new_samples_dict

        for sample_dict in self.samples_dict:
            if sample_dict['LE'] != []:
                for pair in sample_dict['LE']:
                    self.pairs.append(pair)
            if sample_dict['RE'] != []:
                for pair in sample_dict['RE']:
                    self.pairs.append(pair)
        num_pair = len(self.pairs)
        new_idx = np.random.permutation(np.arange(num_pair))
        permutated_pairs = [self.pairs[idx] for idx in new_idx]
        self.pairs = permutated_pairs

    def select_pairs(self, pairs):
        labels = np.array([x[1] for x in pairs])
        hist = AMDDataAREDS.calculate_histogram(labels)
        num_zero, num_non_zero = self._num_zero_non_zero(hist)
        selected_pairs = []
        if len(pairs) <= self.args.min_participant_pairs:
            selected_pairs = pairs
            return selected_pairs
        if num_zero == 0:
            selected_pairs = pairs
            return selected_pairs
        if num_non_zero == 0:
            if self.args.min_participant_pairs == 0:
                return selected_pairs
            else:
                selected_idx = self.random_select(np.arange(len(pairs)), self.args.min_participant_pairs)
                selected_pairs = [pairs[i] for i in selected_idx]
                return selected_pairs
        else:
            idx_non_zero = np.where(labels != 0)[0]
            idx_zero = np.where(labels == 0)[0]
            if num_non_zero >= (self.args.min_participant_pairs / 2):
                selected_idx_zero = self.random_select(idx_zero, num_non_zero)
                selected_pairs = [pairs[i] for i in idx_non_zero] + [pairs[j] for j in selected_idx_zero]
                return selected_pairs
            else:
                selected_idx_zero = self.random_select(idx_zero, self.args.min_participant_pairs - num_non_zero)
                selected_pairs = [pairs[i] for i in idx_non_zero] + [pairs[j] for j in selected_idx_zero]
                return selected_pairs

    def random_select(self, idx, num_sample):
        """samples num_sample indices from range(0, max_idx)."""
        if idx.shape[0] <= num_sample:
            return idx
        selected_idx = np.random.choice(idx, num_sample, replace=False)
        return selected_idx

    def _num_zero_non_zero(self, hist):
        labels = list(hist.keys())
        num_zero, num_non_zero = 0, 0
        if '0' in labels:
            num_zero = hist['0']
        if '1' in labels:
            num_non_zero += hist['1']
        if '2' in labels:
            num_non_zero += hist['2']

        return num_zero, num_non_zero


class AMDDataAREDS(pl.LightningDataModule):
    def __init__(self, args):
        super(AMDDataAREDS, self).__init__()
        self.args = args
        self.full_dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        start = time.time()
        samples_dict = self.prepare_image_names()
        self.full_dataset = AMDDatasetAREDS(samples_dict, self.args)  # TODO: No transforms yet.
        np.random.seed(self.args.seed_data)
        self.full_dataset.prepare_pairs(balanced=self.args.balanced)
        logging.warning('Time to prepare the data: {:.4f}'.format(time.time() - start))

    def setup(self, stage=None):
        start = time.time()
        num_participant = len(self.full_dataset.samples_dict)
        num_train, num_val, num_test = self.data_split_numbers(num_participant, self.args.data_split_ratio)
        np.random.seed(self.args.seed_data)
        idx = np.arange(num_participant)
        idx_train = np.random.choice(idx, num_train, replace=False)
        idx = np.setxor1d(idx, idx_train)
        idx_val = np.random.choice(idx, num_val, replace=False)
        idx_test = np.setxor1d(idx, idx_val)
        data_transforms = None
        if self.args.transform:
            data_transforms = self.prepare_transforms(self.args.transform)
        self.train_set = AMDDatasetAREDS([self.full_dataset.samples_dict[idx] for idx in idx_train],
                                         self.args, transform=data_transforms)
        self.val_set = AMDDatasetAREDS([self.full_dataset.samples_dict[idx] for idx in idx_val],
                                       self.args, transform=data_transforms)
        self.test_set = AMDDatasetAREDS([self.full_dataset.samples_dict[idx] for idx in idx_test],
                                        self.args, transform=data_transforms)
        self.train_set.prepare_pairs(), self.val_set.prepare_pairs(), self.test_set.prepare_pairs()
        logging.warning('Time to partition the data: {:.4f}'.format(time.time() - start))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.args.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)

    def prepare_image_names(self):
        samples_file = ('FullVisitsGap{}Thr{}.pkl'.format(self.args.visit_gap, self.args.late_AMD_threshold))
        samples_file_path = os.path.join(self.args.data_dir, samples_file)
        if os.path.exists(samples_file_path):
            with open(samples_file_path, 'rb') as fname:
                samples = pickle.load(fname)
                fname.close()
            return samples
        else:
            samples = []
            pheno_file_lines = self.read_pheno_file()
            file_names = [x[0] for x in pheno_file_lines[1:]]
            amd_sev = [x[9] for x in pheno_file_lines[1:]]
            amd_sev_dict = {u: v for u, v in zip(file_names, amd_sev)}
            del file_names, amd_sev

            with open(self.args.image_dict, 'r') as fname:
                samples_dict = json.load(fname)
                fname.close()
            sample_names = list(samples_dict.keys())
            for name in sample_names:
                sample_dict = samples_dict[name]
                sample_visits = sample_dict['VISIT']
                sample = self._prepare_sample(name, sample_visits, self.args.visits, amd_sev_dict)
                samples.append(sample)
            with open(samples_file_path, 'wb') as fname:
                pickle.dump(samples, fname)
                fname.close()
            return samples

    def _prepare_sample(self, name, sample_visits, target_visits, label_dict):
        late_AMD_th = self.args.late_AMD_threshold
        sample = {'ID': name, 'LE': [], 'RE': []}
        sample_visits_no = list(sample_visits.keys())
        # sample_visits_no = [x if x!= 'BL' else '00' for x in sample_visits_no] # TODO: a few samples have 'BL' instead of '00'
        label_dict_keys = list(label_dict.keys())
        for i in range(len(target_visits)):  # Preparing left eye pairs.
            if (target_visits[i][0] in sample_visits_no) and (target_visits[i][1] in sample_visits_no):
                curr_visit, next_visit = target_visits[i][0], target_visits[i][1]
                curr_batch, next_batch = sample_visits[curr_visit]['LE']['LS']['Batch'], \
                                         sample_visits[next_visit]['LE']['LS']['Batch']
                curr_img, next_img = sample_visits[curr_visit]['LE']['LS']['Img_name'], \
                                     sample_visits[next_visit]['LE']['LS']['Img_name']
                if (curr_batch == 'nan') or (next_batch == 'nan') or (curr_img == 'nan') or (next_img == 'nan'):
                    continue
                curr_img_dng, next_img_dng = '', ''
                if curr_batch == '2010':
                    curr_img_dng = curr_img[:-3] + 'dng'
                if next_batch == '2010':
                    next_img_dng = next_img[:-3] + 'dng'
                if not ((curr_img in label_dict_keys) or (curr_img_dng in label_dict_keys)):
                    if self.args.verbose > 1:
                        logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t LE'.
                                        format(curr_img, curr_img_dng, curr_batch, name))
                    continue
                if not ((next_img in label_dict_keys) or (next_img_dng in label_dict_keys)):
                    if self.args.verbose > 1:
                        logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t LE'.
                                        format(next_img, next_img_dng, next_batch, name))
                    continue

                else:
                    if curr_batch == '2014':
                        curr_label = label_dict[curr_img]
                    else:
                        curr_label = label_dict[curr_img_dng]
                    if next_batch == '2014':
                        next_label = label_dict[next_img]
                    else:
                        next_label = label_dict[next_img_dng]
                    try:
                        curr_label, next_label = int(curr_label), int(next_label)
                        if curr_label <= (late_AMD_th - 1) and next_label <= (late_AMD_th - 1):
                            y = 0
                        elif curr_label <= (late_AMD_th - 1) and next_label >= late_AMD_th:
                            y = 1
                        elif curr_label >= late_AMD_th and next_label >= late_AMD_th:
                            y = 2
                        else:
                            if self.args.verbose > 1:
                                logging.warning('Inconsistent case: name {} \t curr_img {}:{} - {} \t next_img {}:{} - '
                                                '{}'.format(name, curr_img, curr_label, curr_batch, next_img,
                                                            next_label, next_batch))
                            continue
                        curr_img = os.path.join(self.args.image_dir[curr_batch], curr_img)
                        next_img = os.path.join(self.args.image_dir[next_batch], next_img)
                        if not os.path.exists(curr_img):
                            if self.args.verbose > 1:
                                logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t LE \t VisitNO:'
                                                ' {}'.format(curr_img, curr_batch, name, curr_visit))
                            continue
                        if not os.path.exists(next_img):
                            if self.args.verbose > 1:
                                logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t LE \t VisitNO:'
                                                ' {}'.format(next_img, next_batch, name, next_visit))
                            continue
                        sample['LE'].append(((curr_img, next_img), y))
                    except:
                        continue
            else:
                continue

        for i in range(len(target_visits)):  # Preparing Right eye pairs.
            if (target_visits[i][0] in sample_visits_no) and (target_visits[i][1] in sample_visits_no):
                curr_visit, next_visit = target_visits[i][0], target_visits[i][1]
                curr_batch, next_batch = sample_visits[curr_visit]['RE']['LS']['Batch'], \
                                         sample_visits[next_visit]['RE']['LS']['Batch']
                curr_img, next_img = sample_visits[curr_visit]['RE']['LS']['Img_name'], \
                                     sample_visits[next_visit]['RE']['LS']['Img_name']
                if (curr_batch == 'nan') or (next_batch == 'nan') or (curr_img == 'nan') or (next_img == 'nan'):
                    continue
                curr_img_dng, next_img_dng = '', ''
                if curr_batch == '2010':
                    curr_img_dng = curr_img[:-3] + 'dng'
                if next_batch == '2010':
                    next_img_dng = next_img[:-3] + 'dng'
                if not ((curr_img in label_dict_keys) or (curr_img_dng in label_dict_keys)):
                    if self.args.verbose > 1:
                        logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t RE'.
                                        format(curr_img, next_img_dng, curr_batch, name))
                    continue
                if not ((next_img in label_dict_keys) or (next_img_dng in label_dict_keys)):
                    if self.args.verbose > 1:
                        logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t RE'.
                                        format(next_img, next_img_dng, next_batch, name))
                    continue

                else:
                    if curr_batch == '2014':
                        curr_label = label_dict[curr_img]
                    else:
                        curr_label = label_dict[curr_img_dng]
                    if next_batch == '2014':
                        next_label = label_dict[next_img]
                    else:
                        next_label = label_dict[next_img_dng]
                    try:
                        curr_label, next_label = int(curr_label), int(next_label)
                        if curr_label <= (late_AMD_th - 1) and next_label <= (late_AMD_th - 1):
                            y = 0
                        elif curr_label <= (late_AMD_th - 1) and next_label >= late_AMD_th:
                            y = 1
                        elif curr_label >= late_AMD_th and next_label >= late_AMD_th:
                            y = 2
                        else:
                            if self.args.verbose > 1:
                                logging.warning('Inconsistent case: name {} \t curr_img {}:{} - {} \t next_img {}:{} -'
                                                ' {}'.format(name, curr_img, curr_label, curr_batch, next_img,
                                                             next_label, next_batch))
                            continue
                        curr_img = os.path.join(self.args.image_dir[curr_batch], curr_img)
                        next_img = os.path.join(self.args.image_dir[next_batch], next_img)
                        if not os.path.exists(curr_img):
                            if self.args.verbose > 1:
                                logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t LE \t VisitNO:'
                                                ' {}'.format(curr_img, curr_batch, name, curr_visit))
                            continue
                        if not os.path.exists(next_img):
                            if self.args.verbose > 1:
                                logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t LE \t VisitNO:'
                                                ' {}'.format(next_img, next_batch, name, next_visit))
                            continue
                        sample['RE'].append(((curr_img, next_img), y))
                    except:
                        continue
            else:
                continue
        return sample

    def read_pheno_file(self):
        pheno_file_name = self.args.pheno_dir
        if not os.path.isfile(pheno_file_name):
            raise ValueError('There is no file with name "' + self.args.phenotype_file +
                             f'" in the {self.args.data_dir} directory.')
        else:
            file = open(pheno_file_name, "r")
            lines = file.readlines()
            file.close()
            output_lines = []
            for i in range(len(lines)):
                if self.args.verbose > 0 and i % 10000 == 0:
                    if self.args.verbose > 1:
                        logging.warning(f'Reading phenotype file line #{i}')
                output_lines.append(lines[i].split('\t'))
                if output_lines[-1][-1][-1] == '\n':
                    output_lines[-1][-1] = output_lines[-1][-1][:-1]
            return output_lines

    def class_weights(self):
        train_hist = self.data_histogram('train')
        hist_vals = torch.tensor(list(train_hist.values()), dtype=torch.float)
        hist_vals = hist_vals / hist_vals.sum()
        weights = 1 / hist_vals
        weights = weights / weights.sum()
        return weights

    def data_histogram(self, partition='whole', log=False):
        """
        :param log: Whether logging the statistics of partition or not.
        :param partition:'whole' stands for the entire dataset. options: 'whole', 'train', 'val', 'test'.
        :return: histogram of the labels of the desired partition.
        """
        if partition == 'whole':
            labels = np.array([p[1] for p in self.full_dataset.pairs])
        elif partition == 'train':
            labels = np.array([p[1] for p in self.train_set.pairs])
        elif partition == 'val':
            labels = np.array([p[1] for p in self.val_set.pairs])
        elif partition == 'test':
            labels = np.array([p[1] for p in self.test_set.pairs])
        else:
            raise NotImplementedError

        hist = self.calculate_histogram(labels, log=log)
        return hist

    @staticmethod
    def calculate_histogram(y, log=False):
        hist = {}
        unique_labels = np.unique(y)
        for l in unique_labels:
            hist[f'{int(l)}'] = ((y == l).sum())
        if log:
            logging.warning(hist.keys())
            logging.warning(hist.values())
            logging.warning((list(hist.values()))/sum(list(hist.values())))
        return hist

    @staticmethod
    def data_split_numbers(num_data, split_ratio):
        """
        :param num_data: Total number of data.
        :param split_ratio: [ratio_train, ratio_val, ratio_test]
        :return: The number of data in each split.
        """
        num_train = int(np.ceil(num_data * split_ratio[0]))
        num_val = int(np.ceil(num_data * split_ratio[1]))
        num_test = int(num_data - num_train - num_val)
        return num_train, num_val, num_test

    def prepare_transforms(self, transform):
        if transform[0] == 'resize':
            new_size = self.args.im_resize_shape
            if new_size != 224:
                data_transforms = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor()])
            else:
                data_transforms = transforms.Compose([transforms.ToTensor()])
            return data_transforms
        else:
            raise NotImplementedError('Not implemented {} transform(s) yet.'.format(transform))

    def convert_to_binary(self):
        """
        This function is used when we want to train the network to predict the next step's label based on only the next
        step's label. In the default setting, when we prepare the data, we use both current step's label and the next
        step's one (3 classes), but we only use the one for the next step here (2 classes/binary classification).
        """
        for i in range(len(self.train_set.pairs)):
            if self.train_set.pairs[i][1]:  # If the label is not 0, make it 1.
                self.train_set.pairs[i] = (self.train_set.pairs[i][0], 1)
        for i in range(len(self.val_set.pairs)):
            if self.val_set.pairs[i][1]:
                self.val_set.pairs[i] = (self.val_set.pairs[i][0], 1)
        for i in range(len(self.test_set.pairs)):
            if self.test_set.pairs[i][1]:
                self.test_set.pairs[i] = (self.test_set.pairs[i][0], 1)

    def resize_2014_images_deepseenet(self):
        """
        Performs center cropping preprocessing as done in the DeepSeeNet 'https://github.com/ncbi-nlp/DeepSeeNet' code
        on the original large images in the AREDS dataset for the batch 2014 images.
        """
        img_2014_dir = self.args.image_dir['original_2014']
        img_2014_resized_dir_deepseenet = self.args.image_dir['2014']
        images = os.listdir(img_2014_dir)
        not_loaded = []
        if not os.path.exists(img_2014_resized_dir_deepseenet):
            os.mkdir(img_2014_resized_dir_deepseenet)
        for i, img in enumerate(images):
            if os.path.exists(os.path.join(img_2014_resized_dir_deepseenet, img)):
                logging.warning('Already resized: \t {}/{}: \t {}'.format(i, len(images), img))
                continue
            try:
                curr_img = crop2square(image.load_img(os.path.join(img_2014_dir, img))).resize((224, 224))
                fn = img
                curr_img.save(os.path.join(img_2014_resized_dir_deepseenet, fn))
                logging.warning('saved \t {}/{}: \t {}'.format(i, len(images), img))
            except:
                not_loaded.append(img)
        logging.warning('***********************************************************************')
        logging.warning('Could not load the following images:')
        for i, img_name in enumerate(not_loaded):
            logging.warning('Unable to load: \t {}/{}: \t {}'.format(i, len(not_loaded), img_name))

    def resize_2010_images_deepseenet(self):
        """
        Performs center cropping preprocessing as done in the DeepSeeNet 'https://github.com/ncbi-nlp/DeepSeeNet' code
        on the original large images in the AREDS dataset for the batch 2010 images.
        """
        img_2010_dir = self.args.image_dir['original_2010']
        img_2010_resized_dir_deepseenet = self.args.image_dir['2010']
        images = os.listdir(img_2010_dir)
        not_loaded = []
        if not os.path.exists(img_2010_resized_dir_deepseenet):
            os.mkdir(img_2010_resized_dir_deepseenet)
        for i, img in enumerate(images):
            if os.path.exists(os.path.join(img_2010_resized_dir_deepseenet, img)):
                logging.warning('Already resized: \t {}/{}: \t {}'.format(i, len(images), img))
                continue
            try:
                curr_img = crop2square(image.load_img(os.path.join(img_2010_dir, img))).resize((224, 224))
                fn = img
                curr_img.save(os.path.join(img_2010_resized_dir_deepseenet, fn))
                logging.warning('saved \t {}/{}: \t {}'.format(i, len(images), img))
            except:
                not_loaded.append(img)

        logging.warning('***********************************************************************')
        logging.warning('Could not load the following images:')
        for i, img_name in enumerate(not_loaded):
            logging.warning('Unable to load: \t {}/{}: \t {}'.format(i, len(not_loaded), img_name))

    def resize_ukb_RE_images_deepseenet(self):
        ukb_dir = self.args.image_dir['ukb_original_RE']
        ukb_dir_resized_deepseenet = self.args.image_dir['ukb_RE']
        images = os.listdir(ukb_dir)
        not_loaded = []
        if not os.path.exists(ukb_dir_resized_deepseenet):
            os.mkdir(ukb_dir_resized_deepseenet)
        for i, img in enumerate(images):
            if os.path.exists(os.path.join(ukb_dir_resized_deepseenet, img)):
                logging.warning('Already resized: \t {}/{}: \t {}'.format((i+1), len(images), img))
                continue
            try:
                curr_img = crop2square(image.load_img(os.path.join(ukb_dir, img))).resize((224, 224))
                fn = img
                curr_img.save(os.path.join(ukb_dir_resized_deepseenet, fn))
                logging.warning('saved \t {}/{}: \t {}'.format((i+1), len(images), img))
            except:
                not_loaded.append(img)

        logging.warning('***********************************************************************')
        logging.warning('Could not load the following images:')
        for i, img_name in enumerate(not_loaded):
            logging.warning('Unable to load: \t {}/{}: \t {}'.format((i+1), len(not_loaded), img_name))

    def resize_ukb_LE_images_deepseenet(self):
        ukb_dir = self.args.image_dir['ukb_original_LE']
        ukb_dir_resized_deepseenet = self.args.image_dir['ukb_LE']
        images = os.listdir(ukb_dir)
        not_loaded = []
        if not os.path.exists(ukb_dir_resized_deepseenet):
            os.mkdir(ukb_dir_resized_deepseenet)
        for i, img in enumerate(images):
            if os.path.exists(os.path.join(ukb_dir_resized_deepseenet, img)):
                logging.warning('Already resized: \t {}/{}: \t {}'.format((i+1), len(images), img))
                continue
            try:
                curr_img = crop2square(image.load_img(os.path.join(ukb_dir, img))).resize((224, 224))
                fn = img
                curr_img.save(os.path.join(ukb_dir_resized_deepseenet, fn))
                logging.warning('saved \t {}/{}: \t {}'.format((i+1), len(images), img))
            except:
                not_loaded.append(img)

        logging.warning('***********************************************************************')
        logging.warning('Could not load the following images:')
        for i, img_name in enumerate(not_loaded):
            logging.warning('Unable to load: \t {}/{}: \t {}'.format((i+1), len(not_loaded), img_name))


def crop2square(img):
    """
    Borrowed from DeepSeeNet paper's code. 'https://github.com/ncbi-nlp/DeepSeeNet/blob/master/deepseenet/utils.py
    """
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))

