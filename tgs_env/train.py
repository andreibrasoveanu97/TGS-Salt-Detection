import tensorflow as tf
import tensorflow.contrib.eager as tfe
from metrics import dice_coef
import numpy as np
# tf.compat.v1.enable_eager_execution()


def gradient(model, inputs, targets, loss=tf.keras.losses.binary_crossentropy):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss_val = loss(outputs, targets)
    return tape.gradient(loss_val, model.variables), loss_val


def train(model,
          optimizer: tf.train.Optimizer,
          train_loader: tf.data.Dataset,
          loss=tf.keras.losses.binary_crossentropy,
          metric=dice_coef,
          epochs=100,
          batch_size=20,
          callback=None):
    print('Enter in train')
    train_loader = train_loader.batch(batch_size)
    # train_loader = train_loader.repeat(epochs)
    print('Loader init')
    for epoch in range(epochs):
        print('Begin epoch {}'.format(epoch))
        epoch_losses = []
        for i, batch in enumerate(tfe.Iterator(train_loader)):
            print('batch no {}'.format(i))
            grads, _loss = gradient(model, inputs=batch[0], targets=batch[1], loss=loss)
            optimizer.apply_gradients(zip(grads, model.variables))
            epoch_losses.append(_loss.numpy())
        print('train loss {}'.format(np.mean(epoch_losses)))
