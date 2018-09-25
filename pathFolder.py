#!/usr/bin/env python3.5
#


import argparse
import csv
import os
import glob
import sys


def main(argv):
    file = open("./"+"file.tsv", "w+")
    file.write("id\timage\tcluster\n")
    index = 1

    path = './images/'
    list_directory = os.listdir(path)
    list_directory.sort()

    for in_folders in list_directory:
        list_files = os.listdir(path + in_folders)
        list_files.sort()

        for in_files in list_files:
            if in_files.endswith('.png'):
                name = in_files.split('_')
                file.write("%d\t%s/%s\t%s\n" % (index, in_folders,in_files, name[0]))
                index = index + 1

    file.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))