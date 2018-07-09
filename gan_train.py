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
mnist = input_data.read_data_sets("MNIST_data/")

def lrelu(x):
    return tf.maximum(0.2*x, x)

def upsample_and_concat(x1, size, output_channels, in_channels, batch_size, name):
    output_shape = [batch_size, 2*size, 2*size, output_channels]
    pool_size = 2
    deconv_filter = tf.get_variable(name, [pool_size, pool_size, output_channels, in_channels], initializer=tf.truncated_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, output_shape , strides=[1, pool_size, pool_size, 1] )
    return deconv

def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        d_conv1 = slim.conv2d(images, 32, [5,5], activation_fn=lrelu, scope='d_conv1')
        d_pool1 = slim.max_pool2d(d_conv1, [2,2],  scope='d_pool1')
        d_conv2 = slim.conv2d(d_pool1, 64, [3,3], activation_fn=lrelu, scope='d_conv2')
        d_pool2 = slim.max_pool2d(d_conv2, [2,2], scope='d_pool2')
        d_conv3 = slim.conv2d(d_pool2, 128, [3,3], activation_fn=lrelu, scope='d_conv3')
        d_pool3 =slim.max_pool2d(d_conv3, [2,2], scope='d_pool3')
        d_flatten = slim.flatten(d_pool3)
        d_fc1 = slim.fully_connected(d_flatten, 1024, activation_fn=lrelu, scope='d_fc1')
        output = slim.fully_connected(d_fc1, 1, activation_fn=None, scope='d_output')
        
        return output


def generator(z, batch_size, z_dim):
    size = 7
    dims = 128
    g_fc1 = slim.fully_connected(z, size*size*dims, activation_fn=None, scope='g_fc1')
    g_fc1_reshape = tf.reshape(g_fc1, [-1, size, size, dims])
    g1 = tf.contrib.layers.batch_norm(g_fc1_reshape, epsilon=1e-5, scope='g_bn1')
    g1 = lrelu(g1)

    g1_up = upsample_and_concat(g1, size, int(dims/2), dims, batch_size, name='g_up_covn1')
    g1_up = tf.contrib.layers.batch_norm(g1_up, epsilon=1e-5, scope='g_bn2')
    g1_up = lrelu(g1_up)
    g2_up = upsample_and_concat(g1_up, 2*size, 1, int(dims/2), batch_size, name='g_up_covn2')
    g2_up = tf.contrib.layers.batch_norm(g2_up, epsilon=1e-5, scope='g_bn3')
    g2_up = lrelu(g2_up)
    return tf.sigmoid(g2_up)


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
batch_size = 64

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

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

#reuse the Variable

tf.get_variable_scope().reuse_variables()
sess = tf.Session()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
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
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 100 == 0:
        print("steps:",i)
        print("dLossReal:",dLossReal,"dLossFake:",dLossFake)
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)

saver = tf.train.Saver()
saver.save(sess,'pretrained-model/pretrained_gam.ckpt')
z_batch = np.random.normal(0, 1, size=[10, z_dimensions])
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')

generated_images = generator(z_placeholder, 10, z_dimensions)
images = sess.run(generated_images, {z_placeholder: z_batch})
#for i in range(10):
#    plt.imshow(images[i].reshape([28, 28]), cmap='Greys')
#    plt.show()
