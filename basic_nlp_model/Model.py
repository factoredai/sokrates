import tensorflow as tf
from tensorflow import keras


class Model():
    def __init__(self, specs: dict):
        self.train_data = specs['train_data']
        self.val_data = specs['val_data']
        self.max_tokens = specs['max_tokens']
        self.embedding_dimension = specs['embedding_dimension']
        self.lstm_layers = specs['lstm_layers']
        self.lstm_activation = specs['lstm_activation']
        self.dense_layers = specs['dense_layers']
        self.dense_activation = specs['dense_activation']
        self.batch_size = specs['batch_size']
        self.epochs = specs['epochs']

        self.optimizer = specs['optimizer']

        # Optimizer and callbaks must be passed as eval-ready strings:
        self.optimizer = eval(specs['optimizer'])
        self.callbacks = eval(specs.get('callbacks', '[]'))

        self.numeric_features = [
            'n_lists',
            'n_links',
            'n_tags',
            'num_question_marks',
            'wh_word_count',
            'sentence_count',
            'word_count',
            'example_count',
            'n_linebreaks',
            'title_word_count',
            'title_question_marks'
        ]

        self.tokenizer = specs.get('tokenizer', None)
        self.normalizer = specs.get('normalizer', None)

        self.build_model()

    def build_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = keras.layers.experimental.\
                preprocessing.TextVectorization(self.max_tokens)
            self.tokenizer.adapt(self.train_data['title'].to_list())

    def build_normalizer(self):
        if not self.normalizer:
            self.normalizer = keras.layers.experimental.\
                preprocessing.Normalization()
            self.normalizer.adapt(
                self.train_data[self.numeric_features].to_numpy()
            )

    def build_model(self):
        self.build_tokenizer()
        self.build_normalizer()

        embedding = keras.layers.Embedding(
            self.max_tokens,
            self.embedding_dimension
        )

        # Recurrent part:
        rnn = keras.Sequential([
            self.tokenizer,
            embedding,

            *[
                keras.layers.Bidirectional(keras.layers.LSTM(
                    units,
                    activation=self.lstm_activation,
                    return_sequences=True
                ))
                for units in self.lstm_layers[:-1]
            ],

            keras.layers.Bidirectional(keras.layers.LSTM(
                units=self.lstm_layers[-1],
                activation=self.lstm_activation
            ))
        ])

        # Dense part:
        dnn = keras.Sequential([
            *[
                keras.layers.Dense(
                    units,
                    activation=self.dense_activation
                )
                for units in self.dense_layers
            ],

            keras.layers.Dense(1)
        ])

        # Define inputs:
        inputs = [
            keras.Input((1,), dtype=tf.string),
            keras.Input((11,), dtype=tf.float32)
        ]

        # Define output:
        output = rnn(inputs[0])
        output = tf.concat([output, inputs[1]], axis=1)
        output = dnn(output)

        self.model = keras.Model(inputs, output)

        self.model.compile(
            optimizer=self.optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

    def fit(self):
        X = [
            self.train_data['title'].to_numpy(),
            self.train_data[self.numeric_features].to_numpy()
        ]

        Y = self.train_data['y'].to_numpy()

        X_val = [
            self.val_data['title'].to_numpy(),
            self.val_data[self.numeric_features].to_numpy()
        ]

        Y_val = self.val_data['y'].to_numpy()

        self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, Y_val),
            callbacks=self.callbacks
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y, batch_size=self.batch_size)
