#!/usr/bin/env python3
#


import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy
import pandas
from sklearn import (decomposition, manifold, pipeline)


ncomp = 0


def data_model(name, n):

    if n is None:
        return decomposition.PCA()
    else:
        if name == 'TSNE':
            return manifold.TSNE(random_state=0)
        elif name == 'TSNE-PCA':
            tsne = manifold.TSNE(random_state=0, perplexity=50, early_exaggeration=6.0)
            pca = decomposition.PCA(n_components = n)
            return pipeline.Pipeline([('reduce_dims', pca), ('tsne', tsne)])
        elif name == 'PCA':
                return decomposition.PCA(n_components = n)


def data_info(data, model):
    transformed = [d.split(',') for d in data['features']]

    x_data = numpy.asarray(transformed).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    vis_data = model.fit_transform(x_data)

    # print(len(model.explained_variance_ratio_))

    cont_values = numpy.float32(0)
    cont_layer = 0
    for p in range(len(model.explained_variance_ratio_)):
        aux_info = numpy.float32(model.explained_variance_ratio_[p])
             
        if cont_values <= numpy.float32(0.995):
            cont_values = cont_values + aux_info
            cont_layer = cont_layer + 1
        
        # print(cont_layer)
        # print(cont_values)
        # print(aux_info)
        # print("\n")

    global ncomp
    ncomp = cont_layer

    print("%d %f %d" % (len(model.explained_variance_ratio_), cont_values, cont_layer))

    return


def data_process(data, model):
    transformed = [d.split(',') for d in data['features']]

    x_data = numpy.asarray(transformed).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    vis_data = model.fit_transform(x_data)

    results = []
    for i in range(0, len(data)):
        results.append({'id': data['id'][i], 'x': vis_data[i][0], 'y': vis_data[i][1], 'cluster': data['cluster'][i]})
    return results


def main(argv):
    parser = argparse.ArgumentParser(prog='TSNE')
    parser.add_argument('source', help='path to the source metadata file')
    parser.add_argument('model', help='use named model')
    parser.add_argument('-n', '--components', type=int, default=50, help='components PCA & TSNE-PCA')
    args = parser.parse_args()

    source_file = args.source
    model_name = args.model
    num_components = args.components

    try:
        data = pandas.read_csv(source_file, sep='\t')

        info_model = data_model('PCA', None)
        data_info(data, info_model)

        model_data = data_model(model_name, ncomp)
        results = data_process(data, model_data)

        for i in range(len(results)):
            item = results[i]
            if item['cluster'] == 'F':
                plt.scatter(float(item['x']), float(item['y']), marker='*', color="red")
            elif item['cluster'] == 'ME':
                plt.scatter(float(item['x']), float(item['y']), marker='*', color="green")
            elif item['cluster'] == 'MS':
                plt.scatter(float(item['x']), float(item['y']), marker='*', color="blue")

        red_star = mlines.Line2D([], [], color='red', marker='*', label='F')
        green_star = mlines.Line2D([], [], color='green', marker='*', label='ME')
        blue_star = mlines.Line2D([], [], color='blue', marker='*', label='MS')

        plt.legend(handles=[red_star, green_star, blue_star])
        plt.title("%s %s %d" % (os.path.splitext(source_file)[0], model_name, ncomp))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(False)
        plt.savefig("%s_%s_%d.png" % (os.path.splitext(source_file)[0], model_name, ncomp))
        #plt.show()

    except EnvironmentError as e:
        sys.stderr.write('error: {}\n'.format(e))
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
