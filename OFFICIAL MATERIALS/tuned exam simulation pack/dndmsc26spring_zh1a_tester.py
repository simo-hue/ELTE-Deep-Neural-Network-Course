
#
# Automatic tests for Coding test, variant A (21apr2026) in ELTE IK, Deep Network Development course, 2026 spring
#
# Authors: Imre Molnar, Viktor Varga
#

import json
import requests
import datetime
import platform
import getpass
import copy as copy_module
import os
import urllib
import numpy as np

import torch
import torch.nn as nn

class Tester:

    # Tests

    def __test_dataset_load(self, *args):
        '''
        Expected parameters:
            dataset1: ndarray(n_juices_in_dataset1, n_features) of float32
            dataset2: ndarray(n_juices_in_dataset2, n_features) of float32
        '''
        assert len(args) == 2, "Tester error: __test_dataset_load() expects 2 parameters: dataset1, dataset2. "
        dataset1, dataset2 = args

        try:
            assert type(dataset1) == np.ndarray, "Tester: 'dataset1' should be a numpy array."
            assert type(dataset2) == np.ndarray, "Tester: 'dataset2' should be a numpy array."
            assert dataset1.dtype == np.float32, "Tester: 'dataset1' array should have float32 data type."
            assert dataset2.dtype == np.float32, "Tester: 'dataset2' array should have float32 data type."
            expected_shape1 = (1832, 14)
            expected_shape2 = (874, 14)
            assert dataset1.shape == expected_shape1, f"Tester: 'dataset1' array should have a shape of {expected_shape1}."
            assert dataset2.shape == expected_shape2, f"Tester: 'dataset2' array should have a shape of {expected_shape2}."

            assert not np.any(np.isnan(dataset1)), f"Tester: There should be no NaN (Not-a-Number) values in the 'dataset1' array."
            assert not np.any(np.isnan(dataset2)), f"Tester: There should be no NaN (Not-a-Number) values in the 'dataset2' array."

            assert np.fabs(np.sum(dataset1) - 482915.2) < 10., f"Tester: There might be some problems with the content of your 'dataset1' array. Try to check against the .csv file!"
            assert np.fabs(np.sum(dataset2) - 230474.88) < 10., f"Tester: There might be some problems with the content of your 'dataset2' array. Try to check against the .csv file!"

        finally:
            self.dataset1 = np.copy(dataset1)
            self.dataset2 = np.copy(dataset2)
        
        print("Tester: Dataset loading OK")


    def __test_dataset_merge(self, *args):
        '''
        Expected parameters:
            dataset: ndarray(n_juices, n_features) fl32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_merge() expects 1 parameter: dataset. "
        dataset, = args
        assert (self.dataset1 is not None) and (self.dataset2 is not None), "Tester error: Run tester for task 'A' first."

        try:
            assert type(dataset) == np.ndarray, "Tester: 'dataset' must be a numpy array."
            assert dataset.dtype == np.float32, "Tester: 'dataset' array must have float32 data type."
            assert not np.any(np.isnan(dataset)), "Tester: There should be no NaN values in the 'dataset' array."

            expected_shape_merged = (2706, 14)
            assert dataset.shape == expected_shape_merged, f"Tester: 'dataset' array should have a shape of {expected_shape_merged}."

            dataset_part1 = dataset[:1832,:]
            dataset_part2 = dataset[1832:,:]

            assert np.allclose(dataset_part1, self.dataset1), f"Tester: There might be some problems with the content of your 'dataset' array."
            assert np.allclose(dataset_part2[:,4:], self.dataset2[:,4:]), f"Tester: There might be some problems with the content of your 'dataset' array."
            assert np.allclose(dataset_part2[:,0], self.dataset2[:,3]), f"Tester: There might be some problems with the content of your 'dataset' array."
            assert np.allclose(dataset_part2[:,1], self.dataset2[:,0]), f"Tester: There might be some problems with the content of your 'dataset' array."
            assert np.allclose(dataset_part2[:,2], self.dataset2[:,2]), f"Tester: There might be some problems with the content of your 'dataset' array."
            assert np.allclose(dataset_part2[:,3], self.dataset2[:,1]), f"Tester: There might be some problems with the content of your 'dataset' array."

        finally:
            self.dataset = np.copy(dataset)
        
        print("Tester: Dataset merging OK")


    def __test_dataset_separate_categories(self, *args):
        '''
        Expected parameters:
            dataset_budget: ndarray(n_budget_juices, n_input_features) fl32
            dataset_premium: ndarray(n_premium_juices, n_input_features) fl32

        '''
        assert len(args) == 2, "Tester error: __test_dataset_separate_categories() expects 2 parameters: dataset_budget, dataset_premium. "
        dataset_budget, dataset_premium = args
        assert self.dataset is not None, "Tester error: Run tester for task 'A' & 'B' first."

        try:
            assert type(dataset_budget) == np.ndarray, "Tester: 'dataset_budget' must be a numpy array."
            assert type(dataset_premium) == np.ndarray, "Tester: 'dataset_premium' must be a numpy array."
            assert dataset_budget.dtype == np.float32, "Tester: 'dataset_budget' array must have float32 data type."
            assert dataset_premium.dtype == np.float32, "Tester: 'dataset_premium' array must have float32 data type."

            expected_shape_budget = (1444, 11)
            expected_shape_premium = (1262, 11)
            assert dataset_budget.shape[1] == expected_shape_budget[1], f"Tester: 'dataset_budget' array should have {expected_shape_budget[1]} columns."
            assert dataset_budget.shape == expected_shape_budget, f"Tester: 'dataset_budget' array should have a shape of {expected_shape_budget}."
            assert dataset_premium.shape[1] == expected_shape_premium[1], f"Tester: 'dataset_premium' array should have {expected_shape_premium[1]} columns."
            assert dataset_premium.shape == expected_shape_premium, f"Tester: 'dataset_premium' array should have a shape of {expected_shape_premium}."
                        
            assert np.fabs(np.sum(dataset_budget[:,0]) == 14868.752) < 10., f"Tester: There might be some problems with the content of your 'dataset_budget' array."
            assert np.fabs(np.sum(dataset_premium[:,0]) == 12249.107) < 10., f"Tester: There might be some problems with the content of your 'dataset_premium' array."

        finally:
            self.dataset_budget = np.copy(dataset_budget)
            self.dataset_premium = np.copy(dataset_premium)
        
        print("Tester: Dataset separate categories OK")


    def __test_dataset_split(self, *args):
        '''
        Expected parameters:
            dataset_budget_train, dataset_budget_val, dataset_budget_test: ndarray(n_split_samples, n_input_features) of float32
            dataset_premium_train, dataset_premium_val, dataset_premium_test: ndarray(n_split_samples, n_input_features) of float32
        '''
        assert len(args) == 6, "Tester error: __test_dataset_split() expects 6 parameters: dataset_budget_train, dataset_budget_val, "+\
                                            "dataset_budget_test, dataset_premium_train, dataset_premium_val, dataset_premium_test."
        assert (self.dataset_budget is not None) and (self.dataset_premium is not None), "Tester error: Run tester for task 'A' & 'B' & 'C' first."

        for arr in args:
            assert type(arr) == np.ndarray, "Tester: All of the objects passed to the tester should be NumPy arrays."
            assert arr.ndim == 2, "Tester: All of the arrays passed to the tester should have 2 dimensions."

        dataset_budget_train, dataset_budget_val, dataset_budget_test, dataset_premium_train, dataset_premium_val, dataset_premium_test = args

        # assert split sizes correct
        assert np.fabs(dataset_budget_train.shape[0]/0.7 - dataset_budget_val.shape[0]/0.15) < 30, "Tester: 'dataset_budget_train' or 'dataset_budget_val' split sizes may be incorrect."
        assert np.fabs(dataset_budget_train.shape[0]/0.7 - dataset_budget_test.shape[0]/0.15) < 30, "Tester: 'dataset_budget_train' or 'dataset_budget_test' split sizes may be incorrect."
        assert np.fabs(dataset_premium_train.shape[0]/0.7 - dataset_premium_val.shape[0]/0.15) < 30, "Tester: 'dataset_premium_train' or 'dataset_premium_val' split sizes may be incorrect."
        assert np.fabs(dataset_premium_train.shape[0]/0.7 - dataset_premium_test.shape[0]/0.15) < 30, "Tester: 'dataset_premium_train' or 'dataset_premium_test' split sizes may be incorrect."

        # assert combined shape unchanged
        concat_splits = np.concatenate([dataset_budget_train, dataset_budget_val, dataset_budget_test], axis=0)
        assert concat_splits.shape == self.dataset_budget.shape, "Tester: The concatenated 'dataset_budget_*' splits should match the 'dataset_budget' array produced in the previous task."

        concat_splits = np.concatenate([dataset_premium_train, dataset_premium_val, dataset_premium_test], axis=0)
        assert concat_splits.shape == self.dataset_premium.shape, "Tester: The concatenated 'dataset_premium_*' splits should match the 'dataset_premium' array produced in the previous task."

        print("Tester: Dataset split OK")


    def __test_bincl_iter(self, *args):
        '''
        Expected parameters:
            dataloader_bincl_train, dataloader_bincl_val, dataloader_bincl_test: Iterable objects
        '''
        assert len(args) == 3, "Tester error: __test_bincl_iter() expects 3 parameters: " +\
                                    "dataloader_bincl_train, dataloader_bincl_val, dataloader_bincl_test. "
        dataloader_bincl_train, dataloader_bincl_val, dataloader_bincl_test = args

        dataloader_iters = [dataloader_bincl_train, dataloader_bincl_val, dataloader_bincl_test]
        dataloader_names = ['dataloader_bincl_train', 'dataloader_bincl_val', 'dataloader_bincl_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
          dataloader_iter = copy_module.copy(dataloader_iter)
          for r in dataloader_iter:   # extracting new iterator from DataLoader
            break
          assert len(r) == 2, f"Tester: 'The iterator created by {dataloader_name}' should return two tensors in each iteration."
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor, f"Tester: 'The iterator created by {dataloader_name}' should return tensors."
          assert r0.dtype == r1.dtype == torch.float32, f"Tester: 'The iterator created by {dataloader_name}' should return tensors with float32 data type."
          assert r0.ndim == r1.ndim == 2, f"Tester: The iterator created by '{dataloader_name}' returns tensors with an incorrect shape."
          batch_size, n_feat = r0.shape
          assert r1.shape[0] == batch_size, f"Tester: The iterator created by '{dataloader_name}' returns tensors with an inconsistent shape."
          assert n_feat == 11, f"Tester: Input feature tensor returned by the iterator of '{dataloader_name}' have incorrect shape."
          assert r1.shape[1] == 1, f"Tester: Label tensor returned by the iterator of '{dataloader_name}' have incorrect shape."

          r1_npy = r1.numpy()
          u_r1_npy = np.unique(r1_npy)
          assert set(u_r1_npy) <= set([0., 1.]), f"Tester: Labels returned by the iterator of '{dataloader_name}' might be incorrect."
          if len(r1_npy) >= 32 and (set(u_r1_npy) < set([0., 1.])):
            print(f"Tester: Warning! Are you sure the iterators return samples simultaneously from both categories? - If you see this message often, there might be a problem with your code.\n")

        print("Tester: Dataset iterators for binary classification task OK")


    def __test_bincl_model_learning(self, *args):
        '''
        Expected parameters:
            test_bce, test_acc: float
        '''
        assert len(args) == 2, "Tester error: __test_bincl_model_learning() expects 2 parameters: test_bce, test_acc. "
        test_bce, test_acc = args
        MIN_BCE, MAX_BCE = 0.25, 0.55
        MIN_ACC, MAX_ACC = 0.75, 0.89

        assert MIN_BCE < test_bce < MAX_BCE, f"Tester: A well trained binary classification model should provide a test BCE value between {MIN_BCE} and {MAX_BCE}."
        assert MIN_ACC < test_acc < MAX_ACC, f"Tester: A well trained binary classification model should provide a test accuracy between {MIN_ACC} and {MAX_ACC}."

        print("Tester: Binary classification model learning OK")


    # Utilities

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}
        dataset_content1: None OR str; dataset loaded into a string
        dataset_content2: None OR str; dataset loaded into a string

        # STORED DATA FROM PREVIOUS TESTS
        dataset1: ndarray(n_juices_in_dataset1, n_features) of float32
        dataset2: ndarray(n_juices_in_dataset2, n_features) of float32
        dataset: ndarray(n_juices, n_features) fl32
        dataset_budget: ndarray(n_budget_juices, n_input_features) fl32
        dataset_premium: ndarray(n_premium_juices, n_input_features) fl32

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''
    def __init__(self):

        self.LOAD_DATASET_FROM_LOCAL = True
        self.TESTS_DICT = {
              'dataset_load': self.__test_dataset_load,
              'dataset_merge': self.__test_dataset_merge,
              'dataset_separate_categories': self.__test_dataset_separate_categories,
              'dataset_split': self.__test_dataset_split,
              'bincl_iter': self.__test_bincl_iter,
              #'bincl_model_architecture': self.__test_bincl_model_architecture,
              'bincl_model_learning': self.__test_bincl_model_learning}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}
        self.dataset_content1 = None
        self.dataset_content2 = None
        self.dataset1 = None
        self.dataset2 = None
        self.dataset = None
        self.dataset_budget = None
        self.dataset_premium = None

    def get_dataset_contents(self):
        '''
        Sets self.dataset_content.
        Returns:
            dataset_content1, dataset_content2: str, str
        '''
        with open("zh_files/orangejuice1_dnd26spring_zh_a.csv", 'r') as f:
            self.dataset_content1 = f.read()
        with open("zh_files/orangejuice2_dnd26spring_zh_a.csv", 'r') as f:
            self.dataset_content2 = f.read()
        
        return self.dataset_content1, self.dataset_content2

    def test(self, test_name, *args):
        '''
        Parameters:
            test_name: str
            *args: varargs; the arguments for the selected test
        '''
        if test_name not in self.TESTS_DICT:
            self.print_tester_error_msg(f"Tester error: Invalid test name: {test_name}", "<pre-test>")

        self.test_results[test_name] = False
        succ = False
        test_func = self.TESTS_DICT[test_name]
        try:
            test_func(*args)
        except:
            raise
        else:
            succ = True
        finally:
            self.test_results[test_name] = succ

    def print_tester_error_msg(self, msg, test_name):
        print(f"\nTester error in test '{test_name}' ->  {msg}\n")