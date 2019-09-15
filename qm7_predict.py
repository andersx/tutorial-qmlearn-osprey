#!/usr/bin/env python3

import sys

import pickle
import glob

import numpy as np

import qml
from qml import qmlearn

import sklearn.pipeline
import sklearn.model_selection


np.random.seed(666)

def parse_data():

    filenames = sorted(glob.glob("data/qm7/*.xyz"))[:200]
    data = qmlearn.Data(filenames)
    energies = np.loadtxt("data/hof_qm7.txt", usecols=1)[:200]
    rescaled_energies = qmlearn.preprocessing.AtomScaler().fit_transform(data.nuclear_charges, energies)
    data.set_energies(rescaled_energies)

    return data


def pipeline():


    data = parse_data()
    
    model = sklearn.pipeline.Pipeline(
            [
                ('preprocess', qmlearn.preprocessing.AtomScaler(data)),
                ('representations', qmlearn.representations.CoulombMatrix()),
                ('kernel', qmlearn.kernels.LaplacianKernel()),
                ('model', qmlearn.models.KernelRidgeRegression())
            ],
        )

    training_indices = np.arange(100)
    model.fit(training_indices)
    
    test_indices = np.arange(100,200)
    score = model.score(test_indices)

    print(score)

if __name__ == "__main__":

    pipeline()

    
