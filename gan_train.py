import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

import tensorflow.contrib.slim as slim
'''
import the data
need to change the path
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/")

def lrelu(x):
    return tf.maximun(0.2*x, x)

def discriminator(images):
    d_conv1 = slim.conv2d(images, 32, [5,5], rate=1, activation_fn=lrelu, scope='d_conv1')
    d_pool1 = slim.max_pool2d(d_conv1, [2,2], padding='SAME', scope='d_pool1')
    d_conv2 = slim.conv2d(d_pool1, 64, [3,3], rate=1, activation_fn=lrelu, scope='d_conv2')
    d_pool2 = slim.max_pool2d(d_conv2, [2,2], padding='SAME', scope='d_pool2')
    d_flatten = slim.flatten(d_pool2)
    d_fc1 = slim.fully_connected(d_flatten, 1024, activation_fn=lrelu, scope='d_fc1')
    output = slim.fully_connected(d_fc1, 1, activation_fn=None, scope='d_output')
    return output

def generator(z, batch_size, z_dim):
    g_fc1 = slim.fully_connected(z, 3136, activation_fn=None, scope='g_fc1')
    g_fc1_reshape = tf.reshape(g_fc1, [-1,56,56,1])
    g1 = tf.contrib.layers.batch_norm(g_fc1_reshape, epsilon=1e-5, scope='g_bn1')
    g1 = lrelu(g1)

    g_conv1 = slim.conv2d(g1, int(z_dim/2), [3,3], rate=1, activation_fn=lrelu, scope='g_conv1')
    g_conv2 = slim.conv2d(g_conv1, int(z_dim/4), [3,3], rate=1, activation_fn=lrelu, scope='g_conv2')
    g_conv3 = slim.conv2d(g_conv2, 1, [1,1], strides = [1,2,2,1], rate=1, activation_fn=None, scope='g_conv3')

    return tf.sigmoid(g_conv3)


def plot_generated_image():
    z_dimensions = 100
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])
    generated_image_output = generator(z_placeholder, 1, z_dimensions)
    z_batch = np.random.normal(0, 1, [1, z_dimensions])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        generator_image = sess.run(generated_image_output, feed_dict={z_placeholder: z_batch})
        generator_image = generator_image.reshape([28,28])
        plt.imshow(generator_image, cmap="Greys")

z_dimensions = 100
tf.reset_default_graph()
batch_size = 50

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder) 
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz)
# Dg will hold discriminator prediction probabilities for generated images

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dx, tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.Session()
writer = tf.summary.FileWriter(logdir, sess.graph)



sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(10000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)

    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        generated_images = generator(z_placeholder, 1, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        plt.show()

        # Show discriminator's estimate
        im = images[0].reshape([1, 28, 28, 1])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
        print("Estimate:", estimate)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'pretrained-model/pretrained_gan.ckpt')
    z_batch = np.random.normal(0, 1, size=[10, z_dimensions])
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    generated_images = generator(z_placeholder, 10, z_dimensions)
    images = sess.run(generated_images, {z_placeholder: z_batch})
    for i in range(10):
        plt.imshow(images[i].reshape([28, 28]), cmap='Greys')
        plt.show()
