import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy  
import matplotlib.pyplot as plt
from PIL import Image

rng = numpy.random.RandomState(23455)
input = T.tensor4(name='input')

w_shp = (2, 3, 18, 18) 	#[No of fmaps at layer m or no of filters, no of fmaps in prev layer(m-1) ie input layer,filter height,filter width]
w_bound = numpy.sqrt(3 * 9 * 9)

W = theano.shared(numpy.asarray(
		rng.uniform(
			low = -1.0 / w_bound,
			high = 1.0 / w_bound,
			size = w_shp
			),
		dtype = input.dtype),name = 'w')

b_shp = (2,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

conv_out = conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))  # If base is a vector of 2, this will make it (1 X 2 X 1 X 1),0 represents index to be broadcast in b.
f = theano.function([input],output)

"""
a = numpy.random.random((1,5,19,9)) # [mini batch size,no of input fmaps,image height,image width ]
print a.shape
print f(a)
"""

img  = Image.open(open('lena512color.tiff'))
img = numpy.asarray(img,dtype = 'float64') /256

""" The input is a 4D tensor => [no of batch = 1,no of input fmaps = 3 as rgb, height, width] """

i = img.transpose(2,0,1).reshape(1,3,512,512) #Transpose it in order of indiced 2,0,1
filtered = f(i)

print filtered.shape,i.shape # Filterd shape can be calculated by using formula ( (Input - Filter + 2*padding) / stride ) + 1 )

filtered = filtered.reshape(filtered.shape[2],filtered.shape[3],2)
print filtered.shape
plt.imshow(filtered[:,:,0])#,cmap = 'gray')
plt.show()
plt.imshow(filtered[:,:,1],cmap = 'gray')
plt.show()
