import os
import json
import pandas as pd

from Data.Preprocess import join_path
from Data.Split import Splitter


def split():
    dir = '/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data'
    for case in os.listdir(dir):
        case_dir = join_path(dir, case)
        for node in os.listdir(case_dir):
            node_dir = join_path(case_dir, node)

            df = pd.read_csv(join_path(node_dir, 'attributes.csv'))
            # TODO
            node_dict = {}


def split_temp():
    with open('/homes/clwang/Data/LIDC-IDRI-Crops-Norm/all_datalist.json') as f:
        all_data = json.load(f)

    casename = [x['image'] for x in all_data]
    image = [x['image'] for x in all_data]
    mask = [x['mask'] for x in all_data]
    label = [int(x['label']) for x in all_data]

    df = pd.DataFrame({'casename':casename, 'image':image, 'mask':mask, 'label':label})
    splitter = Splitter(df)

    train, val, test = splitter.split_data(seed=19970516)
    splitter.test_ind()

    train_json = [{'image':row['image'], 'mask':row['mask'], 'label':row['label']} for _, row in train.iterrows()]
    val_json = [{'image':row['image'], 'mask':row['mask'], 'label':row['label']} for _, row in val.iterrows()]
    test_json = [{'image':row['image'], 'mask':row['mask'], 'label':row['label']} for _, row in test.iterrows()]

    save_dir = '/homes/rqyu/Projects/MCMD/Data/Split Temp'
    with open(join_path(save_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f)
    with open(join_path(save_dir, 'val.json'), 'w') as f:
        json.dump(val_json, f)
    with open(join_path(save_dir, 'test.json'), 'w') as f:
        json.dump(test_json, f)


if __name__ == '__main__':
    split_temp()
