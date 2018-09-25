#!/usr/bin/env python3
#


import argparse
import csv
import os
import sys
import random
import time
import math
import operator

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy
import pandas
from sklearn import ( decomposition, manifold, pipeline, svm)


def data_model(n):
    return decomposition.PCA(n_components=n)


def name_svm(name):
    if name == 'SVC_ovo':
        return svm.SVC(decision_function_shape='ovo', kernel='linear', C=0.01)
    elif name == 'SVC_ovr':
        return svm.SVC(decision_function_shape='ovr', kernel='linear', C=0.01)


def data_svm(data_tran, data_test, model):

    tran_lab = [0 for x in range(len(data_tran))]
    tran_data = [[0 for x in range(len(data_tran[0])-1)] for y in range(len(data_tran))]

    for i in range(len(data_tran)):
        tran_lab[i] = data_tran[i][-1]
        for j in range(len(data_tran[i])-1):
            tran_data[i][j] = data_tran[i][j]

    test_lab = [0 for x in range(len(data_test))]
    test_data = [[0 for x in range(len(data_test[0]) - 1)] for y in range(len(data_test))]

    for i in range(len(data_test)):
        test_lab[i] = data_test[i][-1]
        for j in range(len(data_test[i])-1):
            test_data[i][j] = data_test[i][j]
            
    classifier_data = model.fit(tran_data, tran_lab)
    data_predic = classifier_data.predict(test_data)
    
    cnf_matrix = confusion_matrix(test_lab, data_predic, labels=['F', 'ME', 'MS'])
    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, numpy.newaxis]
    
    print("\nConfusion Matrix")
    print(cm)
    
    sum = 0
    num = 0
    for i in range(len(cm[0])):
        sum = sum + float(cm[i][i])
        num = num + 1
        
    media = float(sum / num)
    
    cmedia = float(media) * 100
    print("Score: %f, Media: %f\n" % (cmedia, media))

    return media


def data_process(data, model, n):
    transformed = [d.split(',') for d in data['features']]

    x_data = numpy.asarray(transformed).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    vis_data = model.fit_transform(x_data)

    vis_data2 = [[0 for x in range(n)] for y in range(len(vis_data))]

    for j in range(len(vis_data)):
        for k in range(n):
            vis_data2[j][k] = str(vis_data[j][k])

    results = []
    for i in range(len(data)):
        results.append({'id': data['id'][i], 'features': ','.join(vis_data2[i]), 'cluster': data['cluster'][i]})

    return results


def load_dataset(data, n, num_split, training=[], test=[]):

    data_aux = [[0 for x in range(n+1)] for y in range(len(data))]

    for i in range(len(data)):
        item = data[i]
        vals = item['features']
        ls_vals = vals.split(',')

        for j in range(len(ls_vals)):
            data_aux[i][j] = float(ls_vals[j])

        data_aux[i][n] = str(item['cluster'])
        if random.random() < num_split:
            training.append(data_aux[i])
        else:
            test.append(data_aux[i])

    return


def main(argv):
    parser = argparse.ArgumentParser(prog='SVM')
    parser.add_argument('source', help='Imaginet TSV')
    parser.add_argument('-n', '--components', type=int, default=50, help='components PCA')
    parser.add_argument('-c', '--classification', default='SVC_ovo', help='components PCA')
    parser.add_argument('-s', '--split', type=float, default=0.7, help='Neighbors')
    args = parser.parse_args()

    source_file = args.source
    num_components = args.components
    num_split = args.split
    svm_name =  args.classification

    try:
        model_data = data_model(num_components)
        data = pandas.read_csv(source_file, sep='\t')
        results = data_process(data, model_data, num_components)
        svm_data = name_svm(svm_name)

        training = []
        test = []
        load_dataset(results, num_components, num_split, training, test)

        sco = data_svm(training, test, svm_data)

    except EnvironmentError as e:
        sys.stderr.write('error: {}\n'.format(e))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
