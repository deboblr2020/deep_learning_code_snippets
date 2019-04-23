'''
All you need to do is a permutation of the dimensions from NHWC to NCHW (or the contrary).

The meaning of each letter might help understand:

N: number of images in the batch
H: height of the image
W: width of the image
C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)


'''
import tensorflow as tf

images_nhwc = tf.placeholder(tf.float32, [None, 200, 300, 3])  # input batch
out = tf.transpose(x, [0, 3, 1, 2])       # NCHW
print(out.get_shape())  # the shape of out is [None, 3, 200, 300]



images_nchw = tf.placeholder(tf.float32, [None, 3, 200, 300])  # input batch
out = tf.transpose(x, [0, 2, 3, 1])   #NHWC
print(out.get_shape())  # the shape of out is [None, 200, 300, 3]