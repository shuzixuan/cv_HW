import csv
import json
import numpy as np
import os


with open("../datasets/apple/labelmap.csv", 'a', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['background', '0'])
    writer.writerow(['apple', '1'])

res = []

with open("../datasets/apple/sets/test.txt", 'r') as train_file:
    for line in train_file:
        img_path = '../datasets/apple/images/' + line.strip('\n') + '.png'
        img_path = os.path.abspath(img_path)
        data_path = '../datasets/apple/annotations/' + line.strip('\n') + '.csv'
        with open(data_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            head = next(csv_reader)
            data = list(csv_reader)
            if len(data) == 0:
                res.append([img_path, '', '', '', '', ''])
            else:
                for d in data:
                    item, x, y, r, label = d
                    x, y, r = eval(x), eval(y), eval(r)
                    x1 = x - r
                    x2 = x + r
                    y1 = y - r
                    y2 = y + r
                    res.append([img_path, x1, y1, x2, y2, 'apple'])

with open('../datasets/apple/test_annotations.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    for line in res:
        writer.writerow(line)

res = []

with open("../datasets/apple/sets/train.txt", 'r') as train_file:
    for line in train_file:
        img_path = '../datasets/apple/images/' + line.strip('\n') + '.png'
        img_path = os.path.abspath(img_path)
        data_path = '../datasets/apple/annotations/' + line.strip('\n') + '.csv'
        with open(data_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            head = next(csv_reader)
            data = list(csv_reader)
            if len(data) == 0:
                res.append([img_path, '', '', '', '', ''])
            else:
                for d in data:
                    item, x, y, r, label = d
                    x, y, r = eval(x), eval(y), eval(r)
                    x1 = x - r
                    x2 = x + r
                    y1 = y - r
                    y2 = y + r
                    res.append([img_path, x1, y1, x2, y2, 'apple'])

with open('../datasets/apple/train_annotations.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    for line in res:
        writer.writerow(line)