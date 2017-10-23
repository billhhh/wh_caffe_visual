import sys
# Insert path to caffe, change the path accordingly
sys.path.insert(0,"D:/Projects/caffe-windows/caffe/python")
import caffe

#Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('ResNet-50-deploy.prototxt', 
                'D:/Projects/caffemodel/11_12_resnet50_newFood724_b80_iter_70000.caffemodel', caffe.TEST)
print '------------------------------------------------------------------------'
print '-------------------------Network Architecture---------------------------'
print '------------------------------------------------------------------------'
for layername, layerparam in net.params.items():
    print '  Layer Name : {0:>7}, Weight Dims :{1:12} '.format(layername, layerparam[0].data.shape)
print '------------------------------------------------------------------------'