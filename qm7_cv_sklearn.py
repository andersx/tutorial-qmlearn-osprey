#!/usr/bin/env python3

import sys
import pickle
import glob

import numpy as np

import sklearn.pipeline
import sklearn.model_selection

import qml
from qml import qmlearn

def select_model():


    model = pickle.load(open("model.pickle", "rb"))

    # Doing a grid search over hyper parameters
    # including which kernel to use
    params = {'kernel': [qmlearn.kernels.GaussianKernel(), qmlearn.kernels.LaplacianKernel()],
              'kernel__sigma': [10, 100, 1000],
              'model__l2_reg': [1e-8, 1e-6, 1e-4],
             }

    # CV between the first 200 indices
    indices = np.arange(200)

    # Make a grid-search
    grid = sklearn.model_selection.GridSearchCV(model, cv=2, refit=False, param_grid=params)
    grid.fit(indices)
    
    print("Best hyper parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    print("*** End CV examples ***")


if __name__ == "__main__":

    select_model()

    
