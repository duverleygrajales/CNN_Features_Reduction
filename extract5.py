#!/usr/bin/env python3
#


import argparse
import csv
import os
import sys
import time

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dim = 0
nlayer = 0


def data_imgnet(name):
    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'DenseNet':
        return applications.densenet.DenseNet201(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    elif name == 'ResNet50':
        return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'InceptionResNetV2':
        return applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    elif name == 'NASNet':
        return applications.nasnet.NASNetMobile(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))


def intm_layer(name, data):
    print("\n%s Layers" % name)  # Name Model
    print(data.summary())  # Info Model

    print("\n%s Layers" % name)

    dim0 = 0
    list_layers = []
    aux_i = 0
    for i, layer in enumerate(data.layers):
        list_layers.append({'id': i + 1, 'layer': layer.name})

        aux0 = str(layer.output_shape).split(')')
        aux1 = aux0[0].split(',')
        dim0 = int(aux1[-1])

        aux_i = i + 1
        print(aux_i, layer.name, dim0)

    global dim
    dim = dim0

    input_var = input("\nChoose a Layer of %s: " % name)
    ivar = int(input_var)

    global nlayer
    nlayer = ivar

    item = list_layers[ivar - 1]
    print("\n", item['layer'])

    model = Model(inputs=data.input, outputs=data.get_layer(item['layer']).output)

    if ivar == aux_i:
        return model
    else:
        aux = model.output
        aux = GlobalAveragePooling2D()(aux)
        temp_model = Model(inputs=data.input, outputs=aux)
        return temp_model


def features_imgnet(data, model_data):

    data_aux = list(data)
    items = []

    for i in range(len(data_aux)):
        item = data_aux[i]
        img_file = item['image'].split("/")
        print("%s %s" % (item['id'], img_file[1]))

        img_path = './images/' + item['image']
        img = image.load_img(img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model_data.predict(x)[0]
        array_features = np.char.mod('%f', features)
        
        items.append({'id': item['id'], 'features': ','.join(array_features), 'cluster': item['cluster']})

    return items


def write_file(name, data):
    file = open("./" + name + "_" + str(nlayer) + ".tsv", "w+")
    file.write("id\tfeatures\tcluster\n")

    for i in range(len(data)):
        item = data[i]

        file.write("%s\t%s\t%s\n" % (item['id'], item['features'], item['cluster']))

    file.close()


def main(argv):
    parser = argparse.ArgumentParser(prog='Feature extractor')
    parser.add_argument('source', help='Path source metadata file')
    parser.add_argument('model', help='Name pre-trained model')
    args = parser.parse_args()

    source_file = args.source
    model_name = args.model

    try:
        model_data = data_imgnet(model_name)
        data = pandas.read_csv(source_file, sep='\t')

        in_data = data.T.to_dict().values()
        intmlayer = intm_layer(model_name, model_data)
        print(intmlayer.summary())

        rslt = features_imgnet(in_data, intmlayer)
        write_file(model_name, rslt)

    except EnvironmentError as e:
        print(e)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
