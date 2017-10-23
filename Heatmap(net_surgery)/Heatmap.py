import sys
# Insert path to caffe, change the path accordingly
sys.path.insert(0,"D:/Projects/caffe-windows/caffe/python")
import caffe


#Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('D:/Projects/caffe-windows/caffe/models/bvlc_reference_caffenet/deploy.prototxt', 
                'D:/Projects/caffe-windows/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
print '------------------------------------------------------------------------'
print '-------------------------Network Architecture---------------------------'
print '------------------------------------------------------------------------'
for layername, layerparam in net.params.items():
    print '  Layer Name : {0:>7}, Weight Dims :{1:12} '.format(layername, layerparam[0].data.shape)
print '------------------------------------------------------------------------'



net_full_conv = caffe.Net('D:/Projects/caffe-windows/caffe/models/bvlc_reference_caffenet/deploy.prototxt', 
                'D:/Projects/caffe-windows/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)

params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional'.format(conv, conv_params[conv][0].shape)



params = ['fc6', 'fc7', 'fc8']
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]
net_full_conv.save('D:/Projects/caffemodel/net_surgery/bvlc_caffenet_full_conv.caffemodel')



import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# load input and configure preprocessing
im = caffe.io.load_image('images/cat.jpg')
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load('D:/Projects/caffe-windows/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net input and confidence map (probability of the top prediction at each location)
plt.subplot(1, 2, 1)
plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0,281])