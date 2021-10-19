import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# This allow both tensorflow and pytorch to use the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class LiteModel:

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, X, *args, **kwargs):
        return self.predict(X)

    def predict(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        if type(inp) not in [tuple, list]:
            inp = [inp, ]
        for input_det in self.input_details:
            input_index = input_det["index"]
            input_dtype = input_det["dtype"]
            x = inp[input_index].astype(input_dtype)
            self.interpreter.set_tensor(input_index, x)
        self.interpreter.invoke()
        result = []
        for output_det in self.output_details:
            output_index = output_det["index"]
            # output_dtype = output_det["dtype"]
            out = self.interpreter.get_tensor(output_index)
            result.append(out)
        if len(result) == 1:
            return result[0]
        return result


def to_TFLite(model):
    return LiteModel.from_keras_model(model)


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


patch_size = 6
image_size = 112
patches = Patches(patch_size)(tf.zeros((1, 112, 112, 1)))
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


num_patches = (image_size // patch_size) ** 2
projection_dim = 64
transformer_layers = 2
num_heads = 4
mlp_head_units = [512, 256]
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_actor():
    image = keras.layers.Input(shape=(112, 112, 1))
    prev_action = keras.layers.Input(shape=(1,))

    out = layers.Lambda(lambda x: x / 255)(image)

    # Create patches.
    patches = Patches(patch_size)(out)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        encoded_patches = layers.MaxPooling1D(2)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)

    features = layers.concatenate([representation, prev_action])

    features = mlp(features, hidden_units=mlp_head_units, dropout_rate=0.2)

    out = layers.Dense(2, activation="softmax")(features)
    model = keras.models.Model((image, prev_action), out)
    model.compile()
    return model


def build_critic(model_path, from_actor=True):
    image = keras.layers.Input(shape=(112, 112, 1))
    prev_action = keras.layers.Input(shape=(1,))

    out = layers.Lambda(lambda x: x / 255)(image)

    # Create patches.
    patches = Patches(patch_size)(out)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        encoded_patches = layers.MaxPooling1D(2)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)

    features = layers.concatenate([representation, prev_action])

    features = mlp(features, hidden_units=mlp_head_units, dropout_rate=0.2)

    if from_actor:
        out = layers.Dense(2, activation="softmax")(features)
        model = keras.models.Model((image, prev_action), out)
        model.load_weights(model_path)
    # Now replace the last actor layer with the critic output
    out = layers.Dense(1)(features)

    model = keras.models.Model((image, prev_action), out)
    if not from_actor:
        model.load_weights(model_path)
    model.compile()
    return model
