import zipfile
import os
import glob
from tqdm import tqdm
# npz---dir---jieya
def un_zip(file_name):
    """unzip zip file"""
    file_name_new=file_name.split('.')[0]
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name_new):
        pass
    else:
        os.mkdir(file_name_new)
    for names in zip_file.namelist():
        zip_file.extract(names,file_name_new)
    
    zip_file.close()


if __name__ == '__main__':
    path='/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/zq/rife_t/dataset_train'
    for file_name in tqdm(glob.glob(path+'/*')):
        un_zip(file_name)
    
    path='/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/zq/rife_t/dataset_val'
    for file_name in tqdm(glob.glob(path+'/*')):
        un_zip(file_name)

    print('ok')

    