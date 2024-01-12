import os
import json

def inaturalist(root, is_train, year='2018'):
    root = os.path.join(root, 'inaturalist', str(year))
    json_file = os.path.join(root, '{}{}.json'.format('train' if is_train else 'val', year))
    with open(json_file, 'r') as f:
        data = json.load(f)

    # key: info, images, licenses, annotations, categories
    img_path = []
    label = []
    for idx, row in enumerate(data['images']):
        assert row['id'] == data['annotations'][idx]['id']
        file_name = row['file_name']
        category_id = int(data['annotations'][idx]['category_id'])
        assert category_id >= 0
        file_name = os.path.join(root, file_name)

        img_path.append(file_name)
        label.append(category_id)
    return img_path, label

