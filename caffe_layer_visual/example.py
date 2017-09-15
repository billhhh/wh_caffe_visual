from visualize_caffe import *
import sys

# Make sure caffe can be found
sys.path.append('D:/Projects/caffe-windows/caffe/python')

import caffe


# Load model
net = caffe.Net('finetune_googlenet_newFood724_aug_test.prototxt',
                '11_7_finetune_googlenet_newFood724_iter_100000.caffemodel',
                caffe.TEST)

# visualize_weights(net, 'conv1/7x7_s2', filename='conv1.png')
visualize_weights(net, 'inception_5b/pool_proj', filename='')
