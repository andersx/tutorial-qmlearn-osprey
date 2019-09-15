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
                ('kernel', qmlearn.kernels.GaussianKernel()),
                ('model', qmlearn.models.KernelRidgeRegression())
            ],
        )

    pickle.dump(model, open('model.pickle', 'wb'))

    print(model.get_params().keys())

    indices = np.arange(len(data.coordinates))

    with open('idx.csv', 'w') as f:
        for i in indices:
            f.write('%s\n' % i)

    exit()
    
    params = {'kernel': [qmlearn.kernels.GaussianKernel(), qmlearn.kernels.LaplacianKernel()],
              'kernel__sigma': [10, 100, 1000],
              'model__l2_reg': [1e-8, 1e-6, 1e-4],
}

if __name__ == "__main__":

    pipeline()

    
