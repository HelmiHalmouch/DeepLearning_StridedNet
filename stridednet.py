'''
Here is an implimentation of stridednet CNN architecture 

'''

from keras.models import Sequential 
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout 
from keras.layers.normalization import BatchNormalization
from keras import backend as K

print('The packages are imported !!')

# Define the class for stridednet 

class StridedNet:

	@staticmethod
	def build(width, height, depth, classes, reg, init="he_normal"):
		'''	
	    width : Image width in pixels.
	    height : The image height in pixels.
	    depth : The number of channels for the image.
	    classes : The number of classes the model needs to predict.
	    reg : Regularization method.
	    init : The kernel initializer.
		'''
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# construct the sequential model 
		model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",kernel_initializer=init, kernel_regularizer=reg,
			input_shape=inputShape))

		# here we stack two CONV layers on top of each other where
		# each layerswill learn a total of 32 (3x3) filters
		model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))
 
		# increase the number of filters again, this time to 128
		model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))

		# add af fully conntected layer as function of the clases number 
		model.add(Flatten())
		model.add(Dense(512,kernel_initializer=init))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		#Add softmax classifier 
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# print the summary of the model 
		print('The architecture of stridedNet is :')
		model.summary()

		return model 