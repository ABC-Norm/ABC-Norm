
import os
from tqdm import tqdm


def create_coarse_label(class_txt):
    # c: coarse, f: fine
    cf_label = {'f2c': {}, 'c_index': {}, 'c_map': {}}
    with open(class_txt, 'r') as f:
        for line in f:
            index, name = line.split(' ')

            label = int(index) - 1
            c_name = name.split('_')[-1][:-2]

            cf_label['f2c'][label] = c_name
            cf_label['c_index'].setdefault(c_name, None)

    for idx, c_name in enumerate(cf_label['c_index']):
        cf_label['c_index'][c_name] = idx
        cf_label['c_map'].setdefault(idx, [])

    for f_id in cf_label['f2c']:
        c_name = cf_label['f2c'][f_id]
        c_id = cf_label['c_index'][c_name]
        cf_label['c_map'][c_id].append(f_id)

    return cf_label

def cub(root, is_train):
    # image id, image path
    data_path = os.path.join(root, 'fine_grained', 'cub')
    img_txt_file = open(os.path.join(data_path, 'images.txt'))
    # image id, label
    label_txt_file = open(os.path.join(data_path, 'image_class_labels.txt'))
    # image id, train or test (1 or 0)
    train_val_file = open(os.path.join(data_path, 'train_test_split.txt'))

    img_name_list = []
    for line in img_txt_file:
        img_name_list.append(line[:-1].split(' ')[-1])

    # Label 
    label_list = []
    for line in label_txt_file:
        label_list.append(int(line[:-1].split(' ')[-1]) - 1)

    train_test_list = []
    for line in train_val_file:
        train_test_list.append(int(line[:-1].split(' ')[-1]))

    train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
    test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

    if is_train:
        img_path = [os.path.join(data_path, 'images', train_file) for train_file in
                          train_file_list]
        label = [x for i, x in zip(train_test_list, label_list) if i]
        assert len(img_path) == len(label)
    else:
        img_path = [os.path.join(data_path, 'images', test_file) for test_file in
                         test_file_list]
        label = [x for i, x in zip(train_test_list, label_list) if not i]
        assert len(img_path) == len(label)

    class_txt = os.path.join(data_path, 'classes.txt')
    cf_label = create_coarse_label(class_txt)
    # return img_path, label, cf_label
    return img_path, label

if __name__ == '__main__':
    root = '/work/v20180902/dataset'
    img_path, label, cf_label = cub(root, True)

    print(cf_label['f2c'])
    print(cf_label['c_index'])
    print(cf_label['c_map'])
