import pandas as pd
import numpy as np
from model import Nine5MokModel
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange as trange
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
INPUT_PATH = './input/'

train_df = pd.read_csv(INPUT_PATH + 'train50k.csv')

train_imgs = [str(img) + '.png' for img in train_df.index]
train_labels = [np.asarray(json.loads(str_label)) for str_label in train_df.boards]
train_labels = np.stack(train_labels)

sum_labels = np.sum(train_labels, axis=(1, 2, 3))
x_train, x_valid, y_train, y_valid = train_test_split(train_imgs, 
                                                      train_labels, 
                                                      test_size=0.2, stratify=sum_labels, random_state=1337)
with tf.Session() as sess:
    model = Nine5MokModel(sess, (120, 90), 1)
    model.restore_latest()

    EPOCHS = 50
    BATCH_SIZE = 25 
    epochs_tqdm = tqdm(range(EPOCHS))

    def validation_hook():
        valid_loss, valid_accuracy = model.validate(x_valid, y_valid, BATCH_SIZE)
        epochs_tqdm.set_postfix(valid_loss=valid_loss, valid_accuracy=valid_accuracy)

    model.set_train_hook(every_n=100, hook=validation_hook)

    for epoch in epochs_tqdm:
        epochs_tqdm.set_description('epoch {}'.format(epoch))
        mean_loss, mean_accuracy = model.train(x_train, y_train, BATCH_SIZE, 1e-4)
        model.save()
        epochs_tqdm.set_postfix(mean_loss=mean_loss, mean_accuracy=mean_accuracy)

