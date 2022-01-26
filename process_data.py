import os
import shutil
import random
from tqdm import tqdm
from config import output_path, train, validation, test


train_dir = os.listdir(f"{output_path}/train")
test_dir = os.listdir(f"{output_path}/test")

seed = 1
random.seed(seed)
directory = f'{output_path}train/'
train = f'{output_path}data/train/'
validation = f'{output_path}data/val/'
test = f'{output_path}data/test/'


os.makedirs(train+'cat/')
os.makedirs(train+'dog/')

os.makedirs(test+'cat/')
os.makedirs(test+'dog/')

os.makedirs(validation+'cat/')
os.makedirs(validation+'dog/')

test_samples = train_samples = validation_samples = 0

for line in tqdm(train_dir[:]):
    img = line
    label = 1 if line.split('.')[0] == 'cat' else 0
    random_num = random.random()

    if random_num < 0.8:
        location = train
        train_samples += 1
#     elif random_num < 0.9 :
#         location = validation
#         validation_samples += 1
    else:
        location = test
        test_samples += 1

    if int(float(label)) == 0:
        shutil.copy(
            directory+img,
            location+'dog/'+img,
        )
    else:
        shutil.copy(
            directory + img,
            location + 'cat/' + img,
        )

print(train_samples)
print(test_samples)
print(validation_samples)
print(len(os.listdir(f"{output_path}/data/train/dog")),
      len(os.listdir(f"{output_path}/data/train/cat")))
print(len(os.listdir(f"{output_path}/data/test/dog")),
      len(os.listdir(f"{output_path}/data/test/cat")))
print(len(os.listdir(f"{output_path}/data/val/dog")),
      len(os.listdir(f"{output_path}/data/val/cat")))
