from datetime import datetime
from packaging import version

import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

tf.keras.backend.set_floatx('float64')
# Load the TensorBoard notebook extension.

class Model(tf.keras.Model):
    def __init__(self, x, y):
      super(Model, self).__init__()

      self.x_input = tf.constant(x, dtype=tf.float64)
      self.y_input = tf.constant(y, dtype=tf.float64)

      # Layer 1 - hidden layer 2x2
      self.W1 = tf.Variable(tf.random.uniform([2,2], minval=0., maxval=1., dtype=tf.float64), trainable = True, name='W1')
      self.b1 = tf.Variable(tf.random.uniform([2], minval=0., maxval=1., dtype=tf.float64), trainable = True, name='b1')

      #Layer 2 - output layer 2x1
      self.W2 = tf.Variable(tf.random.uniform([2,1], minval=0., maxval=1., dtype=tf.float64), trainable = True, name='W2')
      self.b2 = tf.Variable(tf.random.uniform([1], minval=0., maxval=1., dtype=tf.float64), trainable = True, name='b2')

    def call(self, inputs):
      inputs = tf.constant(inputs, dtype=tf.float64)

      in_neurons_hidden_layer = tf.add(tf.linalg.matmul(inputs, self.W1),self.b1) #x*W+b
      out_neurons_hidden_layer = tf.sigmoid(in_neurons_hidden_layer)

      in_neurons_output_layer = tf.add(tf.linalg.matmul(out_neurons_hidden_layer, self.W2),self.b2) #x*W+b
      out_neurons_output_layer = tf.sigmoid(in_neurons_output_layer)

      return out_neurons_output_layer


import numpy as np
import pandas as pd


#XOR
x = np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]])
y = np.array([0.,0.,1.,1.])

pd.DataFrame({
    'input1':np.vstack(x).T[0],
    'input2': np.vstack(x).T[1],
    'output':y
})

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/gradient_tape/' + current_time + '/train'
summary_writer = tf.summary.create_file_writer(logdir)

model = Model(x, y)
optimizer = tf.optimizers.SGD(learning_rate=0.9)

def loss(outputs_model, targets):
  error = tf.math.subtract(targets,outputs_model)
  return tf.reduce_sum(tf.square(error))

def get_gradient(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model(inputs), targets)
  return tape.gradient(loss_value, [model.W1, model.b1, model.W2, model.b2])

def run_network(inputs, targets, epochs):
  for i in range(epochs):
    grads=get_gradient(model, inputs, targets)
    optimizer.apply_gradients(zip(grads, [model.W1, model.b1, model.W2, model.b2]))
    loss_epoch = loss(model(inputs), model.y_input)
    with summary_writer.as_default():
      tf.summary.scalar('loss', loss_epoch, step=i)
    if i % 100 == 0 :
      print(f"Loss at the epoch {i}: {loss_epoch}")

run_network(x,y,epochs=200)

# %tensorboard --logdir logs
