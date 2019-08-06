from models import UnetModel
import tensorflow as tf
from train import train
from helpers import create_data_set_from_generator, get_train_generator

ds_train = create_data_set_from_generator(get_train_generator('./tgs/train', mask_dir='masks', img_dir='images'),
                                          _types=(tf.float64, tf.float64),
                                          _shapes=(tf.TensorShape([128, 128, 1]), tf.TensorShape([128, 128, 1])))

model = UnetModel()
optimizer = tf.train.AdamOptimizer(0.0002)
train(model, optimizer, ds_train)
