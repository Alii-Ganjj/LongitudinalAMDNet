"""
Adapted version of AMDDataNextStep that does not consider the data to be pair and simply returns samples with their
labels.
"""

import os
import time
import torch.utils.data
import logging
import json
import pickle

import numpy as np
import torchvision.transforms as transforms
import pytorch_lightning as pl

from skimage import io
from torch.utils.data import DataLoader
from PIL import Image


class AMDDatasetCurrentStep(torch.utils.data.Dataset):
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
        pair = self.pairs[item]
        img_name = pair[0]
        image = io.imread(img_name)
        image = Image.fromarray(image)
        sample_image = self.transforms(image)
        label = pair[1]
        return sample_image, label

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

    def balance_dataset(self):
        """
        Randomly removes the samples from the majority class (0) so that the number of samples with label = 0 becomes
        equal to sum of the number of samples for other classes.
        """
        if not self.pairs:
            raise NotImplementedError('Dataset pairs must be prepared before calling this function!')
        pairs_labels = np.array([x[1] for x in self.pairs])
        num_zero, num_one = (np.where(pairs_labels == 0)[0]).shape[0], (np.where(pairs_labels == 1)[0]).shape[0]
        num_two = (np.where(pairs_labels == 2)[0]).shape[0]
        if num_zero <= num_one + num_two:
            return
        else:
            idx_zero, idx_one = (np.where(pairs_labels == 0)[0]), (np.where(pairs_labels == 1)[0])
            idx_two = (np.where(pairs_labels == 2)[0])
            num_remain = num_one + num_two
            np.random.seed(self.args.seed_data)
            idx_remain_zero = np.random.choice(idx_zero, num_remain)
            np.random.seed(self.args.random_seed)
            idx_remain = list(idx_zero[idx_remain_zero]) + list(idx_one) + list(idx_two)
            new_pairs = [self.pairs[idx] for idx in idx_remain]
            self.pairs = new_pairs

    def select_pairs(self, pairs):
        labels = np.array([x[1] for x in pairs])
        hist = AMDDataCurrentStep.calculate_histogram(labels)
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
        if len(labels) > 2:
            print(labels)
            raise NotImplementedError
        num_zero, num_non_zero = 0, 0
        if '0' in labels:
            num_zero = hist['0']
        if '1' in labels:
            num_non_zero += hist['1']

        return num_zero, num_non_zero

    def downsample_pairs(self, ratio=3):
        new_length = int(np.round(len(self.pairs) / ratio))
        idx = np.random.choice(np.arange(len(self.pairs)), new_length, replace=False)
        new_pairs = [self.pairs[i] for i in idx]
        self.pairs = new_pairs

    def remove_corrupted_images_from_pairs(self):
        for idx, sample_dict in enumerate(self.samples_dict):
            idx_corrupt_right, idx_corrupt_left = [], []
            for i, pair in enumerate(sample_dict['LE']):
                try:
                    img = Image.open(pair[0])  # open the image file
                    img.verify()  # verify that it is, in fact an image
                except (IOError, SyntaxError):
                    idx_corrupt_left.append(i)
                    logging.warning('Bad Image: \t {}'.format(pair[0]))
            for j, pair in enumerate(sample_dict['RE']):
                try:
                    img = Image.open(pair[0])  # open the image file
                    img.verify()  # verify that it is, in fact an image
                except (IOError, SyntaxError):
                    idx_corrupt_right.append(j)
                    logging.warning('Bad Image: \t {}'.format(pair[0]))
            self.samples_dict[idx]['LE'] = [i for j, i in enumerate(sample_dict['LE']) if j not in idx_corrupt_left]
            self.samples_dict[idx]['RE'] = [i for j, i in enumerate(sample_dict['RE']) if j not in idx_corrupt_right]


class AMDDataCurrentStep(pl.LightningDataModule):
    def __init__(self, args):
        super(AMDDataCurrentStep, self).__init__()
        self.args = args
        self.full_dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        start = time.time()
        samples_dict = self.prepare_image_names()
        self.full_dataset = AMDDatasetCurrentStep(samples_dict, self.args)
        # self.full_dataset.remove_corrupted_images_from_pairs()
        self.full_dataset.prepare_pairs(balanced=self.args.balanced)
        logging.warning('Time to prepare the data: {:.4f}'.format(time.time() - start))

    def setup(self, stage=None):
        start = time.time()
        num_participant = len(self.full_dataset.samples_dict)
        np.random.seed(self.args.seed_data)
        num_train, num_val, num_test = self.data_split_numbers(num_participant, self.args.data_split_ratio)
        idx = np.arange(num_participant)
        idx_train = np.random.choice(idx, num_train, replace=False)
        idx = np.setxor1d(idx, idx_train)
        idx_val = np.random.choice(idx, num_val, replace=False)
        idx_test = np.setxor1d(idx, idx_val)
        data_transforms = None
        if self.args.transform:
            data_transforms = self.prepare_transforms(self.args.transform)
        self.train_set = AMDDatasetCurrentStep([self.full_dataset.samples_dict[idx] for idx in idx_train], self.args,
                                               transform=data_transforms)
        self.val_set = AMDDatasetCurrentStep([self.full_dataset.samples_dict[idx] for idx in idx_val], self.args,
                                             transform=data_transforms)
        self.test_set = AMDDatasetCurrentStep([self.full_dataset.samples_dict[idx] for idx in idx_test], self.args,
                                              transform=data_transforms)
        self.train_set.prepare_pairs(), self.val_set.prepare_pairs(), self.test_set.prepare_pairs()
        logging.warning('Time to partition the data: {:.4f}'.format(time.time() - start))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.args.batch_size_train, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)

    def prepare_image_names(self):
        samples_file = ('CurrentStepThr{}.pkl'.format(self.args.late_AMD_threshold))
        samples_file_path = os.path.join(self.args.data_dir, samples_file)
        if os.path.exists(samples_file_path):
            with open(samples_file_path, 'rb') as fname:
                samples = pickle.load(fname)
                fname.close()
            return samples
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
            sample = self._prepare_sample(name, sample_visits, amd_sev_dict)
            samples.append(sample)
        with open(samples_file_path, 'wb') as fname:
            pickle.dump(samples, fname)
            fname.close()
        return samples

    def _prepare_sample(self, name, sample_visits, label_dict):
        sample = {'ID': name, 'LE': [], 'RE': []}
        sample_visits_no = list(sample_visits.keys())
        # sample_visits_no = [x if x!= 'BL' else '00' for x in sample_visits_no] # TODO: a few samples have 'BL' instead of '00'
        label_dict_keys = list(label_dict.keys())

        for visit_no in sample_visits_no:  # Preparing left eye images.
            batch = sample_visits[visit_no]['LE']['LS']['Batch']
            curr_img = sample_visits[visit_no]['LE']['LS']['Img_name']
            if (batch == 'nan') or (curr_img == 'nan'):
                continue
            curr_img_dng = ''
            if batch == '2010':
                curr_img_dng = curr_img[:-3] + 'dng'
            if not ((curr_img in label_dict_keys) or (curr_img_dng in label_dict_keys)):
                if self.args.verbose > 1:
                    logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t LE'.
                                    format(curr_img, curr_img_dng, batch, name))
                continue
            else:
                if batch == '2014':
                    curr_label = label_dict[curr_img]
                elif batch == '2010':
                    curr_label = label_dict[curr_img_dng]
                else:
                    logging.warning('Invalid batch value: Image: {} \t Batch: {}'.format(curr_img, batch))
                    continue

                try:
                    curr_label = int(curr_label)
                    if curr_label <= (self.args.late_AMD_threshold - 1):
                        y = 0
                    else:
                        y = 1
                    curr_img = os.path.join(self.args.image_dir[batch], curr_img)
                    if not os.path.exists(curr_img):
                        if self.args.verbose > 1:
                            logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t LE \t VisitNO: {}'.
                                            format(curr_img, batch, name, visit_no))
                        continue
                    sample['LE'].append((curr_img, y))
                except:
                    continue

        for visit_no in sample_visits_no:  # Preparing Right eye images.
            batch = sample_visits[visit_no]['RE']['LS']['Batch']
            curr_img = sample_visits[visit_no]['RE']['LS']['Img_name']
            if (batch == 'nan') or (curr_img == 'nan'):
                continue
            curr_img_dng = ''
            if batch == '2010':
                curr_img_dng = curr_img[:-3] + 'dng'
            if not ((curr_img in label_dict_keys) or (curr_img_dng in label_dict_keys)):
                if self.args.verbose > 1:
                    logging.warning('Image name not found in pheno file: {}/{} Batch: {} \t Sample: {} \t LE'.
                                    format(curr_img, curr_img_dng, batch, name))
                continue
            else:
                if batch == '2014':
                    curr_label = label_dict[curr_img]
                elif batch == '2010':
                    curr_label = label_dict[curr_img_dng]
                else:
                    logging.warning('Invalid batch value: Image: {} \t Batch: {}'.format(curr_img, batch))
                    continue

                try:
                    curr_label = int(curr_label)
                    if curr_label <= (self.args.late_AMD_threshold - 1):
                        y = 0
                    else:
                        y = 1
                    curr_img = os.path.join(self.args.image_dir[batch], curr_img)
                    if not os.path.exists(curr_img):
                        if self.args.verbose > 1:
                            logging.warning('could not find {} file. \t Batch: {} \t Sample: {} \t RE \t VisitNO: {}'.
                                            format(curr_img, batch, name, visit_no))
                        continue
                    sample['RE'].append((curr_img, y))
                except:
                    continue
        return sample

    def read_pheno_file(self):
        pheno_file_name = self.args.pheno_dir
        if not os.path.isfile(pheno_file_name):
            raise ValueError(
                'There is no file with name "' + self.args.phenotype_file + f'" in the {self.args.data_dir}'
                                                                            f' directory.')
        else:
            file = open(pheno_file_name, "r")
            lines = file.readlines()
            file.close()
            # output_lines = np.array([])
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
        labels = None
        if partition == 'whole':
            labels = np.array([p[1] for p in self.full_dataset.pairs])
        elif partition == 'train':
            labels = np.array([p[1] for p in self.train_set.pairs])
        elif partition == 'val':
            labels = np.array([p[1] for p in self.val_set.pairs])
        elif partition == 'test':
            labels = np.array([p[1] for p in self.test_set.pairs])

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
                # self.train_set.pairs[i][1] = 1
                self.train_set.pairs[i] = (self.train_set.pairs[i][0], 1)
        for i in range(len(self.val_set.pairs)):
            if self.val_set.pairs[i][1]:
                self.val_set.pairs[i] = (self.val_set.pairs[i][0], 1)
        for i in range(len(self.test_set.pairs)):
            if self.test_set.pairs[i][1]:
                self.test_set.pairs[i] = (self.test_set.pairs[i][0], 1)
