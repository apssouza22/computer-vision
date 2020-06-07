from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class ConvAutoencoderBuilder:
	"""
	Autoencoder model builder
	"""

	@staticmethod
	def build(width, height, depth, filters=(32, 64), latentDim=16):
		"""
		:param width: Width of the input image in pixels.
		:param height: Height of the input image in pixels.
		:param depth: Number of channels (i.e., depth) of the input volume.
		:param filters A tuple that contains the set of filters for convolution operations.
		By default, this parameter includes both 32 and 64 filters.
		:param latentDim: The number of neurons in our fully-connected (Dense) latent vector.
		By default, if this parameter is not passed, the value is set to 16
		:return: return the autoencoder model
		"""
		# initialize the input shape to be "channels last" along with the channels dimension itself channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		inputs, latent, volumeSize = ConvAutoencoderBuilder.buildEncoder(chanDim, filters, inputShape, latentDim)

		outputs = ConvAutoencoderBuilder.buildDecoder(chanDim, depth, filters, latent, volumeSize)

		# construct our autoencoder model
		autoencoder = Model(inputs, outputs, name="autoencoder")

		# return the autoencoder model
		return autoencoder

	@staticmethod
	def buildDecoder(chanDim, depth, filters, latent, volumeSize):
		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		x = Dense(np.prod(volumeSize[1:]))(latent)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid", name="decoded")(x)
		return outputs

	@staticmethod
	def buildEncoder(chanDim, filters, inputShape, latentDim):
		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs
		# loop over the number of filters
		for f in filters:
			# apply a CONV => RELU => BN operation
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		# flatten the network and then construct our latent vector
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim, name="encoded")(x)
		return inputs, latent, volumeSize
