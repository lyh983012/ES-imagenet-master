import os
import numpy as np
import pandas as pd


def convert(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    subPath = ['train', 'trainlabel.txt']
    save_infotxt = []
    path_subPath = os.path.join(path, subPath[0])
    save_path_subPath = os.path.join(save_path, subPath[0])

    if not os.path.exists(save_path_subPath):
        os.makedirs(save_path_subPath)

    infotxt = os.path.join(path, subPath[1])

    formats = '.npz'

    with open(infotxt, "r") as f:
        fs = f.readlines()
        for info in fs:
            print(info)
            filename, classnum, a, b = info.split()
            savename = filename
            if not os.path.exists(os.path.join(save_path_subPath, savename.split('/')[0])):
                os.makedirs(os.path.join(save_path_subPath, savename.split('/')[0]))
            realname, sub = filename.split('.')
            filename = realname + formats
            filename = os.path.join(path_subPath, filename)

            dy = (254 - int(b)) // 2
            dx = (254 - int(a)) // 2

            data = np.load(filename)

            datapos = data['pos']
            dataneg = data['neg']

            input = np.zeros([2, 8, 256, 256], dtype='int8')

            x = datapos[:, 0] + dx
            y = datapos[:, 1] + dy
            t = datapos[:, 2] - 1
            input[0, t, x, y] = 1

            x = dataneg[:, 0] + dx
            y = dataneg[:, 1] + dy
            t = dataneg[:, 2] - 1
            input[1, t, x, y] = 1

            input = input[:, :, 16:240, 16:240]
            np.savez_compressed(os.path.join(save_path_subPath, savename), input)
            save_infotxt.append([savename, classnum])

    save = pd.DataFrame(np.array(save_infotxt), columns=['pathname', 'classnum'])
    save.to_csv(os.path.join(save_path, subPath[1].replace('txt', 'csv')), index=False, header=False)


if __name__ == '__main__':
    # path = '/home/huanhuan.gao/data/ES_ImageNet/ES-imagenet-0.18/'

    # save_path = '/home/huanhuan.gao/data/ES_ImageNet/ES-imagenet-0.18-convert/'
    path = '/data1/ES-imagenet-0.18/'
    save_path = '/data1/ES-imagenet-0.18-convert/'

    convert(path=path, save_path=save_path)
