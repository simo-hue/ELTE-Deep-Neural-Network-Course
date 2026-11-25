
#
# Automatic tests for Coding test, variant B (12may2026) in ELTE IK, Deep Network Development course, 2026 spring
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
            dataset_noisy: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_load() expects 1 parameters: dataset_noisy. "
        dataset_noisy, = args

        try:
            assert type(dataset_noisy) == np.ndarray, "Tester: 'dataset_noisy' should be a numpy array."
            assert dataset_noisy.dtype == np.float32, "Tester: 'dataset_noisy' array should have float32 data type."

            expected_shape = (2697, 14)
            assert dataset_noisy.shape == expected_shape, f"Tester: 'dataset_noisy' array should have a shape of {expected_shape}."
            assert not np.any(np.isnan(dataset_noisy)), f"Tester: There should be no NaN (Not-a-Number) values in the 'dataset_noisy' array."

            assert np.fabs(np.sum(dataset_noisy) - 709921.75) < 10., f"Tester: There might be some problems with the content of your 'dataset_noisy' array. Try to check against the .csv file!"

        finally:
            self.dataset_noisy = np.copy(dataset_noisy)
        
        print("Tester: Dataset loading OK")


    def __test_dataset_fill_missing(self, *args):
        '''
        Expected parameters:
            dataset: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_fill_missing() expects 1 parameters: dataset. "
        dataset, = args
        assert self.dataset_noisy is not None, "Tester error: Run tester for task 'A' first."

        assert type(dataset) == np.ndarray, "Tester: 'dataset' must be a numpy array."
        assert dataset.dtype == np.float32, "Tester: 'dataset' array must have float32 data type."
        assert not np.any(np.isnan(dataset)), "Tester: There should be no NaN values in the 'dataset' array."

        try:
            expected_shape_op1 = (2697, 14)
            expected_shape_op23 = (2697, 16)
            if dataset.shape == expected_shape_op1:
                print("Tester: Dataset fill missing: Identified solution aiming for option#1...")

                assert np.array_equal(dataset, self.dataset_noisy), "Tester: In case you are trying to solve option#1, the 'dataset' array must stay the same as the 'dataset_noisy' array."

                print("Tester: Dataset fill missing: Option #1 OK")

            elif dataset.shape == expected_shape_op23:
                print("Tester: Dataset fill missing: Identified solution aiming for option#2 or #3...")

                new_vars = dataset[:,[2,3]]
                assert set(np.unique(new_vars)) == {0, 1}, f"Tester: Incorrect creation of new variable columns of the 'dataset' array."

                assert np.allclose(np.sum(new_vars), 4861., atol=1e-2), "Tester: Incorrect values in new variable columns of the 'dataset' array."

                assert np.allclose(dataset[:,:2], self.dataset_noisy[:,:2]), "Tester: Values coming from the 'dataset_noisy' array should have not been modified (Try re-running the previous task once - it might solve the error!)."
                assert np.allclose(dataset[:,4:], self.dataset_noisy[:,2:]), "Tester: Values coming from the 'dataset_noisy' array should have not been modified (Try re-running the previous task once - it might solve the error!)."

                print("Tester: Dataset fill missing: Option #2 or #3 OK")

            else:
                assert False, "Tester: Shape of 'dataset' array is not matching the required shapes of neither option #1, nor option #2 or #3."

        finally:
            self.dataset = np.copy(dataset)
        

    def __test_dataset_split(self, *args):
        '''
        Expected parameters:
            dataset_split_train, dataset_split_val, dataset_split_test: ndarray(n_split_samples, n_input_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_dataset_split() expects 3 parameters: dataset_split_train, dataset_split_val, dataset_split_test."
        assert self.dataset is not None, "Tester error: Run tester for task 'A' & 'B' first."

        for arr in args:
            assert type(arr) == np.ndarray, "Tester: All of the objects passed to the tester should be NumPy arrays."
            assert arr.ndim == 2, "Tester: All of the arrays passed to the tester should have 2 dimensions."

        dataset_split_train, dataset_split_val, dataset_split_test = args

        # assert split sizes correct
        assert dataset_split_train.shape == (2697-600-600, self.dataset.shape[1]), "Tester: 'dataset_split_train' array has an incorrect shape."
        assert dataset_split_val.shape == (600, self.dataset.shape[1]), "Tester: 'dataset_split_val' array has an incorrect shape."
        assert dataset_split_test.shape == (600, self.dataset.shape[1]), "Tester: 'dataset_split_test' array has an incorrect shape."

        concat_splits = np.concatenate([dataset_split_train, dataset_split_val, dataset_split_test], axis=0)
        assert concat_splits.shape == self.dataset.shape, "Tester: The concatenated 'dataset_split_*' arrays should match the 'dataset' array produced in the previous task."

        # Checking whether content changed
        assert np.fabs(np.sum(concat_splits) - np.sum(self.dataset)) < 10., f"Tester: There should not be any changes in the total sum of values in the 'dataset_split_*' arrays together compared to the values of the 'dataset' array produced in the previous task."

        sample_sums_concat = np.sort(np.sum(concat_splits, axis=1))
        sample_sums_orig = np.sort(np.sum(self.dataset, axis=1))
        assert np.allclose(sample_sums_concat, sample_sums_orig, atol=1e-2), "Tester: Shuffling might have been incorrect: it should only change the order of rows."

        print("Tester: Dataset split OK")

    def __test_cl_model_iter(self, *args):
        '''
        Expected parameters:
            dataloader_cl_train, dataloader_cl_val, dataloader_cl_test: Iterable objects
        '''
        assert len(args) == 3, "Tester error: __test_cl_model_iter() expects 3 parameters: dataloader_cl_train, dataloader_cl_val, dataloader_cl_test. "
        dataloader_cl_train, dataloader_cl_val, dataloader_cl_test = args

        dataloader_iters = [dataloader_cl_train, dataloader_cl_val, dataloader_cl_test]
        dataloader_names = ['dataloader_cl_train', 'dataloader_cl_val', 'dataloader_cl_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
          dataloader_iter = copy_module.copy(dataloader_iter)
          for r in dataloader_iter:   # extracting new iterator from DataLoader
            break
          assert len(r) == 2, f"Tester: 'The iterator created by {dataloader_name}' should return two tensors in each iteration."
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor, f"Tester: 'The iterator created by {dataloader_name}' should return tensors."
          assert r0.dtype == torch.float32, f"Tester: 'The iterator created by {dataloader_name}' should return an input feature tensor with float32 data type."
          assert r1.dtype == torch.int64, f"Tester: 'The iterator created by {dataloader_name}' should return a target label tensor with int64 data type."

          assert r0.ndim == 2, f"Tester: The iterator created by '{dataloader_name}' returns input feature tensors with an incorrect shape."
          assert r1.ndim == 1, f"Tester: The iterator created by '{dataloader_name}' returns target label tensors with an incorrect shape."

          n_required_cols = self.dataset.shape[1]-3
          batch_size, n_feat = r0.shape
          assert r1.shape[0] == batch_size, f"Tester:  The iterator created by '{dataloader_name}' returns tensors with an inconsistent shape."
          assert n_feat == n_required_cols, f"Tester:  The iterator created by '{dataloader_name}' returns an input feature tensor with an incorrect shape."

          if torch.unique(r1).shape[0] == 1:
            print("Tester Warning: Did you make sure that batches contain a mix of different orange juice categories (with a high probability)?")

        print("Tester: Dataset iterators for multi-class classification task OK")

    def __test_cl_model_architecture(self, *args):
        '''
        Expected parameters:
            multiclass_cl_model: PyTorch nn.Module
        '''
        assert len(args) == 1, "Tester error: __test_cl_model_architecture() expects 1 parameter: multiclass_cl_model "
        multiclass_cl_model, = args
        
        assert isinstance(multiclass_cl_model, torch.nn.Module), f"Tester: The 'multiclass_cl_model' neural network model must be of a subtype of torch.nn.Module."
        n_params = sum([p.numel() for p in multiclass_cl_model.parameters(recurse=True) if p.requires_grad])
        n_required_cols = self.dataset.shape[1]-3
        assert n_params == (n_required_cols*20 + 20) + (20*3 + 3), f"Tester: Your 'multiclass_cl_model' neural network architecture does not follow the task description (incorrect parameter count)."

        print("Tester: Multi-class classification model architecture OK")

    def __test_cl_model_learning(self, *args):
        '''
        Expected parameters:
            test_ce, test_acc: float
        '''
        assert len(args) == 2, "Tester error: __test_cl_model_learning() expects 2 parameters: test_ce, test_acc. "
        test_ce, test_acc = args
        MIN_CE, MAX_CE = 0.55, 0.9
        MIN_ACC, MAX_ACC = 0.58, 0.75

        assert MIN_CE < test_ce < MAX_CE, f"Tester: A well trained multi-class classification model should provide a test CE value between {MIN_CE} and {MAX_CE}."
        assert MIN_ACC < test_acc < MAX_ACC, f"Tester: A well trained multi-class classification model should provide a test accuracy between {MIN_ACC} and {MAX_ACC}."

        print("Tester: Multi-class classification model learning OK")


    # Utilities

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}
        dataset_content: None OR str; dataset loaded into a string

        # STORED DATA FROM PREVIOUS TESTS
        dataset_noisy: ndarray(n_juices_in_dataset, n_features) of float32
        dataset: ndarray(n_juices, n_features) fl32

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''
    def __init__(self):

        self.LOAD_DATASET_FROM_LOCAL = True
        self.TESTS_DICT = {
              'dataset_load': self.__test_dataset_load,
              'dataset_fill_missing': self.__test_dataset_fill_missing,
              'dataset_split': self.__test_dataset_split,
              'cl_iter': self.__test_cl_model_iter,
              'cl_model_architecture': self.__test_cl_model_architecture,
              'cl_model_learning': self.__test_cl_model_learning}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}
        self.dataset_content = None
        self.dataset_noisy = None
        self.dataset = None

    def get_dataset_content(self):
        '''
        Sets self.dataset_content.
        Returns:
            dataset_content: str
        '''
        with open("zh_files/orangejuice_dnd26spring_zh_b.csv", 'r') as f:
            self.dataset_content = f.read()
            
        return self.dataset_content

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