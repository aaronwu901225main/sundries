import os
from sklearn.model_selection import train_test_split

# dataset root for BraTS 2023 Part 2
train_data_path = "../dataset/brats2023_part_2/data"

ids = sorted([d for d in os.listdir(train_data_path) if d.startswith("BraTS-GLI-")])
print(f"Found {len(ids)} cases under {train_data_path}")

train_ids, val_test_ids = train_test_split(ids, test_size=0.2, random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=21)
print("Using {} images for training, {} images for validation, {} images for testing.".format(
    len(train_ids), len(val_ids), len(test_ids)))

base_dir = "../dataset/brats2023_part_2"
with open(os.path.join(base_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_ids))
with open(os.path.join(base_dir, 'valid.txt'), 'w') as f:
    f.write('\n'.join(val_ids))
with open(os.path.join(base_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_ids))

print("Split files written to:")
print(os.path.join(base_dir, 'train.txt'))
print(os.path.join(base_dir, 'valid.txt'))
print(os.path.join(base_dir, 'test.txt'))
