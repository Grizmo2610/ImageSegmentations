import os
import shutil
import kagglehub

destination_dir = './content/data'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

print(f'{"=" * 25}Dowloading COCO 2017 dataset {"=" * 25}')
path = kagglehub.dataset_download('awsaf49/coco-2017-dataset')
data_path = shutil.move(path, destination_dir)
print(f'Data moved to: {data_path}')
print('Data source import complete.')

print(f'{"=" * 25}Dowloading lvis-v1 2017 dataset {"=" * 25}')
path = lvis_v1_path = kagglehub.dataset_download('alexanderyyy/lvis-v1')
data_path = shutil.move(path, destination_dir)
print(f'Data moved to: {data_path}')
print('Data source import complete.')