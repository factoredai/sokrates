import tensorflow as tf
from tensorflow import keras
from scipy.special import expit


class Model():
    def __init__(self, specs: dict):
        self.train_data = specs['train_data']
        self.val_data = specs['val_data']
        self.max_tokens = specs['max_tokens']
        self.sequence_length = specs['sequence_length']
        self.embedding_dimension = specs['embedding_dimension']
        self.lstm_layers = specs['lstm_layers']
        self.lstm_activation = specs['lstm_activation']
        self.dense_layers = specs['dense_layers']
        self.dense_activation = specs['dense_activation']
        self.batch_size = specs['batch_size']
        self.epochs = specs['epochs']

        self.reg_type = {
            'l1': keras.regularizers.l1,
            'l2': keras.regularizers.l2
        }[specs.get('reg_type', 'l2')]

        self.lstm_reg = specs.get('lstm_reg', [None]*len(self.lstm_layers))
        assert len(self.lstm_reg) == len(self.lstm_layers)
        self.lstm_reg = [
            self.reg_type(reg) if reg else None for reg in self.lstm_reg
        ]

        self.dense_reg = specs.get('dense_reg', [None]*len(self.dense_layers))
        assert len(self.dense_reg) == len(self.dense_layers)
        self.dense_reg = [
            self.reg_type(reg) if reg else None for reg in self.dense_reg
        ]

        self.lstm_dropout = specs.get(
            'lstm_dropout', [None]*len(self.lstm_layers)
        )
        assert len(self.lstm_dropout) == len(self.lstm_layers)
        self.lstm_dropout = [
            keras.layers.Dropout(rate) if rate else None
            for rate in self.lstm_dropout
        ]

        self.dense_dropout = specs.get(
            'dense_dropout', [None]*len(self.dense_layers)
        )
        assert len(self.dense_dropout) == len(self.dense_layers)
        self.dense_dropout = [
            keras.layers.Dropout(rate) if rate else None
            for rate in self.dense_dropout
        ]

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
                preprocessing.TextVectorization(
                    max_tokens=self.max_tokens,
                    output_sequence_length=self.sequence_length
                )
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
        rnn = keras.Sequential([self.tokenizer, embedding])
        for units, reg, dropout in zip(
            self.lstm_layers[:-1], self.lstm_reg[:-1], self.lstm_dropout[:-1]
        ):
            rnn.add(
                keras.layers.Bidirectional(keras.layers.LSTM(
                        units,
                        activation=self.lstm_activation,
                        return_sequences=True,
                        kernel_regularizer=reg,
                        recurrent_regularizer=reg
                ))
            )
            if dropout:
                rnn.add(dropout)

        rnn.add(
            keras.layers.Bidirectional(keras.layers.LSTM(
                units=self.lstm_layers[-1],
                activation=self.lstm_activation,
                kernel_regularizer=self.lstm_reg[-1],
                recurrent_regularizer=self.lstm_reg[-1]
            ))
        )
        if dropout:
            rnn.add(self.lstm_dropout[-1])

        # Dense part:
        dnn = keras.Sequential([
            *[
                keras.layers.Dense(
                    units,
                    activation=self.dense_activation,
                    kernel_regularizer=reg
                )
                for units, reg in zip(self.dense_layers, self.dense_reg)
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

        self.history = self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, Y_val),
            callbacks=self.callbacks
        )

        self.history = self.history.history

    def predict(self, X):
        return expit(self.model.predict(X))

    def evaluate(self):
        X_val = [
            self.val_data['title'].to_numpy(),
            self.val_data[self.numeric_features].to_numpy()
        ]

        Y_val = self.val_data['y'].to_numpy()

        print('\nStarting evaluation on validation set:')

        return self.model.evaluate(
            X_val, Y_val, batch_size=min(len(X_val[1]), 200000)
        )
