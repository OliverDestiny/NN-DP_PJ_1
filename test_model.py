import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN(
    conv_params=[
        {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
    ],
    fc_params=[32 * 28 * 28, 128, 10],
    act_func='ReLU'
)
model.load_model(r'.\best_models\best_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs.reshape(-1, 1, 28, 28))
print(nn.metric.accuracy(logits, test_labs))