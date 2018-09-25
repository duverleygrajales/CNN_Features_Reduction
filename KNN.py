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
from sklearn import (
    decomposition,
    manifold,
    pipeline,
)


def data_model(n):
        return decomposition.PCA(n_components=n)


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


def dist_euclidean(x1, x2, leng):
    dist = 0
    for x in range(leng):
        dist += pow((x1[x] - x2[x]), 2)
    return math.sqrt(dist)


def get_nbr(training, test, k):
    dist = []
    leng = len(test)-1
    for x in range(len(training)):
        dist0 = dist_euclidean(test, training[x], leng)
        dist.append((training[x], dist0))
    dist.sort(key=operator.itemgetter(1))
    nbr = []
    for x in range(k):
        nbr.append(dist[x][0])
    return nbr


def get_resp(nbr):
    cvotes = {}
    for x in range(len(nbr)):
        resp = nbr[x][-1]
        if resp in cvotes:
            cvotes[resp] += 1
        else:
            cvotes[resp] = 1
    sVotes = sorted(cvotes.items(), key=operator.itemgetter(1), reverse=True)
    return sVotes[0][0]


def get_accy(test, pred):
    rslt = 0
    test_lab = [0 for x in range(len(test))]
    
    #print(test)
    #print("******************************************")
    #print(pred)
    
    for i in range(len(test)):
        test_lab[i] = test[i][-1]
        # if test[x][-1] == pred[x]:
        #     rslt += 1
        
    cnf_matrix = confusion_matrix(test_lab, pred, labels=['F', 'ME', 'MS'])
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
            
    return cmedia


def main(argv):
    parser = argparse.ArgumentParser(prog='KNN')
    parser.add_argument('source', help='Imaginet TSV')
    parser.add_argument('-n', '--components', type=int, default=50, help='components PCA')
    parser.add_argument('-k', '--neighbors', type=int, default=5, help='Neighbors')
    parser.add_argument('-s', '--split', type=float, default=0.7, help='Neighbors')
    args = parser.parse_args()

    source_file = args.source
    num_components = args.components
    k = args.neighbors
    num_split = args.split

    try:
        model_data = data_model(num_components)
        data = pandas.read_csv(source_file, sep='\t')
        results = data_process(data, model_data, num_components)
        # print(results)

        training = []
        test = []
        load_dataset(results, num_components, num_split, training, test)

        pred = []
        for x in range(len(test)):
            nbr = get_nbr(training, test[x], k)
            rslt = get_resp(nbr)
            pred.append(rslt)

        accy = get_accy(test, pred)

    except EnvironmentError as e:
        sys.stderr.write('error: {}\n'.format(e))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
