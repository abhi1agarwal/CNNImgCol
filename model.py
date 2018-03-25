
def conv_stack(data, filters, s):
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        #output = BatchNormalization()(output)
        return output



def getmodel():
	# shape: A shape tuple (integer), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors
	embed_input = Input(shape=(1000,))


	#Encoder
	encoder_input = Input(shape=(256, 256, 1,))
	encoder_output = conv_stack(encoder_input, 64, 2)
	encoder_output = conv_stack(encoder_output, 128, 1)
	encoder_output = conv_stack(encoder_output, 128, 2)
	encoder_output = conv_stack(encoder_output, 256, 1)
	encoder_output = conv_stack(encoder_output, 256, 2)
	encoder_output = conv_stack(encoder_output, 512, 1)
	encoder_output = conv_stack(encoder_output, 512, 1)
	encoder_output = conv_stack(encoder_output, 256, 1)

	#Fusion
	fusion_output = RepeatVector(32 * 32)(embed_input) 
	fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
	fusion_output = concatenate([fusion_output, encoder_output], axis=3) 
	fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output) 

	#Decoder
	decoder_output = conv_stack(fusion_output, 128, 1)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = conv_stack(decoder_output, 64, 1)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = conv_stack(decoder_output, 32, 1)
	decoder_output = conv_stack(decoder_output, 16, 1)
	decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
	decoder_output = UpSampling2D((2, 2))(decoder_output)

	model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
	return model
	
