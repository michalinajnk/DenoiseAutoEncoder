from keras import Input
from keras.layers import Conv3D, ReLU, BatchNormalization, Reshape, Conv3DTranspose, Activation
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.python.keras.backend import shape


class Autoencoder:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # width x height x channels
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [3, 5, 3]
        self.latent_space_dim = latent_space_dim # 2 -> bottleneck have 2 dimension

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")


    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv3D_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder' )

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")


    def _add_conv3D_layers(self, input_layer):
        x = input_layer
        for layer_idx in range(self._num_conv_layers):
            x = self._add_conv3D_layer(layer_idx, x)
        return x

    def _add_conv3D_layer(self, idx, x):
        number = idx + 1
        conv_layer = Conv3D(
            filters=self.conv_filters[idx],
            kernel_size=self.conv_kernels[idx],
            strides=self.conv_strides[idx],
            padding="same",
            name=f"encoder_conv_layer_{number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{number}")(x)
        return x

    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = shape(x)
        """MaxPooling"""

    def _add_conv_LSTM_layers(self,input):
        pass

    def _build_decoder(self):
        input = self._add_decoder_input()
        #dense was here but not really for video data
        reshape_input = self._add_reshape_layer(input)
        conv3D_transpose_layer = self._add_conv3D_transpose_layers(reshape_input)
        decoder_output = self._add_decoder_output(conv3D_transpose_layer)
        self.decoder = Model(input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(self.latent_space_dim, name="decoder_input")

    def _add_reshape_layer(self, layer):
        return Reshape(self._shape_before_bottleneck)(layer)


    def _add_conv3D_transpose_layers(self, layer):
        for layer_idx in reversed(range(1, self._num_conv_layers)):
            layer = self._add_conv3D_transpose_layer(layer_idx, layer)
        return layer

    def _add_conv3D_transpose_layer(self,idx, x):
        number = idx + 1
        conv_transpose_layer = Conv3DTranspose(
            filters=self.conv_filters[idx],
            kernel_size=self.conv_kernels[idx],
            strides=self.conv_strides[idx],
            padding="same",
            name=f"decoder_conv_transpose_layer_{number}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{number}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv3DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)


    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size, num_epochs)


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(5, 360, 240, 2),
        conv_filters=(),
        conv_kernels=(),
        conv_strides=(),
        latent_space_dim=2
    )

    autoencoder.summary()