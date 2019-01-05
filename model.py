import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange
from operator import itemgetter
from batchup import data_source
import cv2


class Nine5MokModel(object):
    def __init__(self, sess, imgsize, imgchannel):
         
        if len(imgsize) is not 2 and type(imgsize) is not tuple: 
            raise ValueError('Invalid image size. imgsize param must be tuple that has 2 length')
                   
        if type(imgchannel) is not int:
            raise ValueError('Invalid image channel, imgchannel param must be interger type value')

        self._input_width = imgsize[0]
        self._input_height = imgsize[1]
        self._BOARD_SIZE = 9

        self._sess = sess

        self._global_step = tf.Variable(0, name='global_step', trainable=True)

        self._input = tf.placeholder(tf.float32, shape=(None, self._input_width, self._input_height, imgchannel), name='imgs_input')

        self._is_train = tf.placeholder_with_default(False, shape=(),name='is_train')

        self._labels = tf.placeholder(tf.float32, shape=(None, self._BOARD_SIZE, self._BOARD_SIZE, 2), name='imgs_label')
        
        self._learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        # 120, 90 -> 60, 45
        x = self._inception_conv(self._input, 16) 
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
       
        # 60, 45 -> 30, 22
        x = self._inception_conv(x, 32)
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
       

        # 30, 22 -> 15, 11
        x = self._inception_conv(x, 64)
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
       
        # 15, 11 -> 7, 5 
        x = self._inception_conv(x, 128)
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
        
        # 7, 5 -> 3, 2
        x = self._inception_conv(x, 256)
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)

        
        # 3, 2 -> 1, 1
        x = self._inception_conv(x, 512)
        x = tf.layers.max_pooling2d(x, [2, 2], strides=2)
        x = tf.reshape(x, shape=(-1, 512))
        x = tf.layers.dense(x, 9 * 9 * 2, activation=tf.nn.sigmoid)            
        x = tf.reshape(x, shape=(-1, 9, 9, 2))
        self._logits = x
        self._loss_op = tf.losses.mean_squared_error(self._labels, self._logits, reduction=tf.losses.Reduction.MEAN) 

        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): 
            self._train_op = optimizer.minimize(loss=self._loss_op, global_step=self._global_step)

        acc, self._accuracy_op = tf.metrics.accuracy(labels=self._labels, predictions=tf.round(self._logits), name='accuracy_operation')
        self._loss_summary = tf.summary.scalar('loss', self._loss_op)

        self._accuracy_summary = tf.summary.scalar('accuracy', self._accuracy_op)
        self._train_summary_writer = tf.summary.FileWriter('./logs/tain', graph_def=sess.graph_def)
        self._validation_summary_writer =  tf.summary.FileWriter('./logs/validation', graph_def=sess.graph_def)

        self._hook_every_n = 0
        self._hook = None
        self._saver = tf.train.Saver(max_to_keep=100)

    def _residual_conv(self, input_tensor, output_channel):
        block_input = tf.layers.conv2d(input_tensor, output_channel, [3, 3], strides=1, padding='SAME')
        x = tf.nn.leaky_relu(block_input, alpha=0.1)
        x = tf.layers.batch_normalization(x, training=self._is_train) 
        block_input = tf.layers.batch_normalization(block_input, training=self._is_train)

        # conv_block
        x = tf.layers.conv2d(x, output_channel, [3, 3], padding='SAME')
        x = tf.layers.batch_normalization(x, training=self._is_train)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        x = tf.layers.conv2d(x, output_channel, [3, 3], padding='SAME')
        x = tf.layers.batch_normalization(x, training=self._is_train)

        return tf.add(x, block_input)

    def _inception_conv(self, input_tensor, output_channel):
        if output_channel % 4 is not 0:
            raise ValueError('output_channel must be multiples of 4.')

        one_channel = output_channel // 4 
        
        ## first pipe line
        first1x1conv = tf.layers.conv2d(input_tensor, one_channel, [1, 1], strides=1, padding='SAME')
        first1x1conv = tf.nn.leaky_relu(first1x1conv, alpha=0.1)

        first1x3conv = tf.layers.conv2d(first1x1conv, one_channel, [1, 3], strides=1, padding='SAME')
        first1x3conv = tf.nn.leaky_relu(first1x3conv, alpha=0.1)

        first3x1conv = tf.layers.conv2d(first1x3conv, one_channel, [3, 1], strides=1, padding='SAME')
        first3x1conv = tf.nn.leaky_relu(first3x1conv, alpha=0.1)

        first1x3conv2 = tf.layers.conv2d(first3x1conv, one_channel, [1, 3], strides=1, padding='SAME')
        first1x3conv2 = tf.nn.leaky_relu(first1x3conv2, alpha=0.1)

        first3x1conv2 = tf.layers.conv2d(first1x3conv2, one_channel, [3, 1], strides=1, padding='SAME')
        first3x1conv2 = tf.nn.leaky_relu(first3x1conv2, alpha=0.1)
        first_output = first3x1conv2
        
        ## second pipe line

        second1x1conv = tf.layers.conv2d(input_tensor, one_channel, [1, 1], strides=1, padding='SAME')
        second1x1conv = tf.nn.leaky_relu(second1x1conv, alpha=0.1)


        second1x3conv = tf.layers.conv2d(second1x1conv, one_channel, [1, 3], strides=1, padding='SAME')
        second1x3conv = tf.nn.leaky_relu(second1x3conv, alpha=0.1)
        second3x1conv = tf.layers.conv2d(second1x3conv, one_channel, [3, 1], strides=1, padding='SAME')
        second3x1conv = tf.nn.leaky_relu(second3x1conv, alpha=0.1)
        second_output = second3x1conv
        
        ## third pipe line
        third_pool = tf.layers.max_pooling2d(input_tensor, [3, 3], strides=1, padding='SAME')
        third1x1conv = tf.layers.conv2d(third_pool, one_channel, [1, 1], strides=1, padding='SAME')
        third1x1conv = tf.nn.leaky_relu(third1x1conv, alpha=0.1)
        third_output = third1x1conv
        

        ## fourth pipe line
        fourth1x1conv = tf.layers.conv2d(input_tensor, one_channel, [1, 1], strides=1, padding='SAME')
        fourth1x1conv = tf.nn.leaky_relu(fourth1x1conv, alpha=0.1)
        fourth_output = fourth1x1conv    

        return tf.concat([first_output, second_output, third_output, fourth_output], axis=-1)

        
    def set_train_hook(self, every_n, hook):
        self._hook = hook
        self._hook_every_n = every_n

    def minibatch_preprocessing(self, minibatch_filenames):
        BASE_PATH = './input/train50k_s/'
        img_for_stacking = []
        for filename in minibatch_filenames:
            img = cv2.imread(BASE_PATH + filename, 0)
            img = np.reshape(img, (90, 120, 1))
            img_for_stacking.append(np.transpose(img, axes=[1, 0, 2]))
        return np.stack(img_for_stacking, axis=0)

    def train(self, input_filenames, labels, batch_size, learning_rate):
        input_filename_idxs = np.array([i for i in range(0, len(input_filenames))])

        train_ds = data_source.ArrayDataSource([input_filename_idxs, labels])
        train_batch_iter = train_ds.batch_iterator(batch_size=batch_size)

        train_batch_total = len(input_filenames) // batch_size if len(input_filenames) % batch_size == 0 else len(input_filenames) // batch_size + 1
        
        epoch_train_loss = []
        epoch_train_accuracy = []

        train_batch_tqdm = tqdm(train_batch_iter, total=train_batch_total)
       
        for train_step, [minibatch_filename_idxs, minibatch_train_labels] in enumerate(train_batch_tqdm):
            train_batch_tqdm.set_description('training...')
            
            minibatch_filenames = itemgetter(*minibatch_filename_idxs)(input_filenames)

            minibatch_files = self.minibatch_preprocessing(minibatch_filenames)

            train_feed = {
                self._input: minibatch_files,
                self._labels: minibatch_train_labels,
                self._learning_rate: learning_rate,
                self._is_train: True,
            }

            minibatch_loss, loss_summary, _ = \
            self._sess.run([self._loss_op, self._loss_summary , self._train_op], feed_dict=train_feed)
            
            train_feed = {
                self._input: minibatch_files,
                self._labels: minibatch_train_labels,
            }

            minibatch_accuracy, accuracy_summary = self._sess.run([self._accuracy_op, self._accuracy_summary], feed_dict=train_feed)

            epoch_train_loss.append(minibatch_loss)
            epoch_train_accuracy.append(minibatch_accuracy)

            global_step = self._sess.run(self._global_step)
            
            self._train_summary_writer.add_summary(loss_summary, global_step=global_step)
            self._train_summary_writer.flush()
            self._train_summary_writer.add_summary(accuracy_summary, global_step=global_step)
            self._train_summary_writer.flush()


            train_batch_tqdm.set_postfix(minibatch_loss=minibatch_loss, minibatch_accuracy=minibatch_accuracy)

            if self._hook and global_step % self._hook_every_n == 0:
                self._hook()
        
        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_accuracy = np.mean(epoch_train_accuracy)

        return epoch_train_loss, epoch_train_accuracy

    def validate(self, input_filenames, labels, batch_size):
        input_filename_idxs = np.array([i for i in range(0, len(input_filenames))])

        valid_ds = data_source.ArrayDataSource([input_filename_idxs, labels])
        valid_batch_iter = valid_ds.batch_iterator(batch_size=batch_size)

        valid_batch_total = len(input_filenames) // batch_size if len(input_filenames) % batch_size == 0 else len(input_filenames) // batch_size + 1
        
        epoch_valid_loss = []
        epoch_valid_accuracy = []

        valid_batch_tqdm = tqdm(valid_batch_iter, total=valid_batch_total)
        for valid_step, [minibatch_valid_filename_idxs, minibatch_valid_labels] in enumerate(valid_batch_tqdm):
            valid_batch_tqdm.set_description('validating...')

            minibatch_valid_filenames = itemgetter(*minibatch_valid_filename_idxs)(input_filenames)
            minibatch_valid_files = self.minibatch_preprocessing(minibatch_valid_filenames)

            valid_feed = {
                self._input: minibatch_valid_files,
                self._labels: minibatch_valid_labels,
            }

            minibatch_loss, minibatch_accuracy = \
            self._sess.run([self._loss_op, self._accuracy_op], feed_dict=valid_feed)
            
            epoch_valid_loss.append(minibatch_loss)
            epoch_valid_accuracy.append(minibatch_accuracy)

            valid_batch_tqdm.set_postfix(minibatch_loss=minibatch_loss, minibatch_accuracy=minibatch_accuracy)
    
        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_accuracy = np.mean(epoch_valid_accuracy)
        
        global_step = self._sess.run(self._global_step)

        valid_loss_summary = tf.Summary()
        valid_loss_summary.value.add(tag="loss", simple_value=epoch_valid_loss)
        valid_loss_summary.value.add(tag="accuracy", simple_value=epoch_valid_accuracy)
        self._validation_summary_writer.add_summary(valid_loss_summary, global_step=global_step)
        self._validation_summary_writer.flush()

        return epoch_valid_loss, epoch_valid_accuracy

    def predict(self, input_filenames, batch_size):
        input_filename_idxs = np.array([i for i in range(0, len(input_filenames))])

        predict_ds = data_source.ArrayDataSource([input_filename_idxs])
        predict_batch_iter = predict_ds.batch_iterator(batch_size=batch_size)

        predict_batch_total = len(input_filenames) // batch_size if len(input_filenames) % batch_size == 0 else len(input_filenames) // batch_size + 1
        
        epoch_logits = []

        predict_batch_tqdm = tqdm(predict_batch_iter, total=predict_batch_total)
        for predict_step , [minibatch_predict_filename_idxs] in enumerate(predict_batch_tqdm):
            predict_batch_tqdm.set_description('predicting...')

            minibatch_predict_filenames = itemgetter(*minibatch_predict_filename_idxs)(input_filenames)
            minibatch_predict_files = self.minibatch_preprocessing(minibatch_predict_filenames)

            logit_feed = {
                self._input: minibatch_predict_files,
            }

            minibatch_logits =  self._sess.run(self._logits , feed_dict=logit_feed)
            
            epoch_logits.append(minibatch_logits)
        return np.concatenate(epoch_logits, axis=0)
    
    def save(self):
        self._saver.save(self._sess, './model/nine5mok', global_step=self._global_step, write_meta_graph=False)

    def restore(self, checkpoint):
        self._saver.restore(self._sess, checkpoint)

    def restore_latest(self):
        latest_model = tf.train.latest_checkpoint('./model/')
        if not latest_model:
            self._sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        else:
            self._saver.restore(self._sess, latest_model)
            self._sess.run(tf.local_variables_initializer())

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    Nine5MokModel(sess, (120, 90), 1)



