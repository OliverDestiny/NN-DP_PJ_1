# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import time
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

cnn_model = nn.models.Model_CNN(
    conv_params=[
        {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
    ],
    fc_params=[32 * 28 * 28, 128, 10],
    act_func='ReLU'
)

optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM_CNN(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

time_start = time.time()
runner.train([train_imgs.reshape(-1, 1, 28, 28), train_labs], [valid_imgs.reshape(-1, 1, 28, 28), valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

time_end = time.time()
print('time cost', time_end - time_start, 's')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()