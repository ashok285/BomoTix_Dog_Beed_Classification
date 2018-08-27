import shutil
import os
import pandas as pd

data_dir       = "/home/ashokkunam9/dogbeed/data" 
old_train_path = '/home/ashokkunam9/dogbeed/data/train'
new_train_path = '/home/ashokkunam9/dogbeed/data/for_train'
old_test_path  = '/home/ashokkunam9/dogbeed/data/test'
new_test_path  = '/home/ashokkunam9/dogbeed/data/for_test'
fakeBreed      = '0'

df = pd.read_csv(os.path.join('.', data_dir, 'labels.csv'))
old_train_path = os.path.join('.', data_dir, old_train_path)
new_train_path = os.path.join('.', data_dir, new_train_path)
if os.path.exists(new_train_path):
    shutil.rmtree(new_train_path)
os.makedirs(new_train_path)

for i, (fname, breed) in df.iterrows():
    new_train_path2 = os.path.join('%s', '%s') % (new_train_path, breed)
    if not os.path.exists(new_train_path2):
        os.makedirs(new_train_path2)
    shutil.copy(os.path.join(old_train_path,  '%s.jpg') % fname,
                os.path.join(new_train_path2, '%s.jpg') % fname)
                
                
df = pd.read_csv(os.path.join('.', data_dir, 'sample_submission.csv'))
old_test_path = os.path.join('.', data_dir, old_test_path)
new_test_path = os.path.join('.', data_dir, new_test_path)
if os.path.exists(new_test_path):
    shutil.rmtree(new_test_path)
os.makedirs(new_test_path)

if os.path.exists(os.path.join(new_test_path, fakeBreed)):
    shutil.rmtree(os.path.join(new_test_path, fakeBreed))
os.makedirs(os.path.join(new_test_path, fakeBreed))

for test_file in os.listdir(old_test_path):
    shutil.copy(os.path.join(old_test_path, test_file),
                os.path.join(new_test_path, fakeBreed))           
