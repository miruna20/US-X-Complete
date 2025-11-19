import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import logging
import sys

# in the paper they consider 26 partial scans per mesh
sys.path.append("utils")
from mmcv.ops import three_interpolate, furthest_point_sample, gather_points, grouping_operation

class verse2020_lumbar(data.Dataset):

    def __init__(self, train_path, val_path, test_path, apply_trafo=True, sigma=0.005, Xray_labelmap=True, prefix="train", num_partial_scans_per_mesh=16):
        logging.info("Using vertebrae dataset")
        if prefix == "train":
            self.file_path = train_path

        elif prefix == "val":
            self.file_path = val_path

        elif prefix == "test":
            self.file_path = test_path
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.apply_trafo = apply_trafo
        self.sigma = sigma
        self.Xray_labelmap = Xray_labelmap

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.number_per_classes = np.array(input_file['number_per_class'][()])
        if(self.Xray_labelmap):
            assert 'labelmaps' in input_file
            self.xray_labelmaps = np.array(input_file['labelmaps'][()])
        self.num_partial_scans_per_mesh = num_partial_scans_per_mesh

        print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # get the partial input point cloud based on index
        partial = self.input_data[index]

        if(self.apply_trafo):
            # apply noise
            partial = add_gaussian_noise(partial,self.sigma)

        partial = torch.from_numpy(partial)

        # for the case in which we have multiple incomplete for one GT
        complete = torch.from_numpy((self.gt_data[index // self.num_partial_scans_per_mesh]))
        label = (self.labels[index])

        if(self.Xray_labelmap):
            labelmap = torch.from_numpy(self.xray_labelmaps[index//self.num_partial_scans_per_mesh])
        else:
            labelmap = np.empty(2)

        return label,partial,labelmap,complete


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        logging.info("Using objects dataset")

        if prefix == "train":
            # self.file_path = 'completion/data/MVP_Train_CP.h5'
            # self.file_path = 'completion/data/first250Vertebrae.h5'
            self.file_path = 'completion/data/vertebrae_train_lumbar.h5'

        elif prefix == "val":
            self.file_path = 'completion/data/MVP_Test_CP.h5'
            # self.file_path = 'completion/data/next250Vertebrae.h5'
            # self.file_path = 'completion/data/vertebrae_val_lumbar.h5'

        elif prefix == "test":
            # self.file_path = 'completion/data/MVP_ExtraTest_Shuffled_CP.h5'
            self.file_path = 'completion/data/vertebrae_test_lumbar.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        # if prefix is not "test":
        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        """
        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // nrPartialScansPerMesh]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial
        """
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, partial, complete

def add_gaussian_noise(pcd, sigma):
    """
    Applies gaussian noise with a certain sigma along y direction of the given pcd

    """
    # sample a vector of size is from a gaussian distribution
    individual_points_shifts_y_axis = np.random.normal(loc=0.0, scale=sigma, size=pcd.shape[0])

    # add noise
    pcd[:, 1] += individual_points_shifts_y_axis

    return pcd



