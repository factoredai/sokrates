import tensorflow as tf
from tensorflow import keras
from scipy.special import expit


class Model():
    def __init__(self, specs: dict):
        self.train_data = specs['train_data']
        self.val_data = specs['val_data']
        self.max_tokens = specs['max_tokens']
        self.embedding_dimension = specs['embedding_dimension']
        self.body_sequence_length = specs['body_sequence_length']
        self.body_layers = specs['body_layers']
        self.body_activation = specs.get('body_activation', 'tanh')
        self.title_sequence_length = specs['title_sequence_length']
        self.title_layers = specs['title_layers']
        self.title_activation = specs.get('title_activation', 'tanh')
        self.dense_layers = specs['dense_layers']
        self.dense_activation = specs['dense_activation']
        self.batch_size = specs['batch_size']
        self.epochs = specs['epochs']

        self.reg_type = {
            'l1': keras.regularizers.l1,
            'l2': keras.regularizers.l2
        }[specs.get('reg_type', 'l2')]

        self.body_reg = specs.get('body_reg', [None]*len(self.body_layers))
        assert len(self.body_reg) == len(self.body_layers)
        self.body_reg = [
            self.reg_type(reg) if reg else None for reg in self.body_reg
        ]

        self.title_reg = specs.get('title_reg', [None]*len(self.title_layers))
        assert len(self.title_reg) == len(self.title_layers)
        self.title_reg = [
            self.reg_type(reg) if reg else None for reg in self.title_reg
        ]

        self.dense_reg = specs.get('dense_reg', [None]*len(self.dense_layers))
        assert len(self.dense_reg) == len(self.dense_layers)
        self.dense_reg = [
            self.reg_type(reg) if reg else None for reg in self.dense_reg
        ]

        self.body_dropout = specs.get(
            'body_dropout', [None]*len(self.body_layers)
        )
        assert len(self.body_dropout) == len(self.body_layers)
        self.body_dropout = [
            keras.layers.Dropout(rate) if rate else None
            for rate in self.body_dropout
        ]

        self.title_dropout = specs.get(
            'title_dropout', [None]*len(self.title_layers)
        )
        assert len(self.title_dropout) == len(self.title_layers)
        self.title_dropout = [
            keras.layers.Dropout(rate) if rate else None
            for rate in self.title_dropout
        ]

        self.dense_dropout = specs.get(
            'dense_dropout', [None]*len(self.dense_layers)
        )
        assert len(self.dense_dropout) == len(self.dense_layers)
        self.dense_dropout = [
            keras.layers.Dropout(rate) if rate else None
            for rate in self.dense_dropout
        ]

        # Specs that must be passed as eval-ready strings:
        self.optimizer = eval(specs['optimizer'])
        self.callbacks = eval(specs.get('callbacks', '[]'))
        self.body_layer_type = eval(specs.get(
            'body_layer_type', "keras.layers.LSTM"
        ))
        self.title_layer_type = eval(specs.get(
            'title_layer_type', "keras.layers.LSTM"
        ))

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
        self.max_sequence_length = max(
            self.body_sequence_length, self.title_sequence_length
        )

        if (not self.tokenizer) or \
                self.tokenizer.get_config()['max_tokens'] != \
                self.max_tokens or \
                self.tokenizer.get_config()['output_sequence_length'] != \
                self.max_sequence_length:

            self.tokenizer = keras.layers.experimental.\
                preprocessing.TextVectorization(
                    max_tokens=self.max_tokens,
                    output_sequence_length=self.max_sequence_length
                )
            self.tokenizer.adapt(
                self.train_data['title'].to_list() +
                self.train_data['body'].to_list()
            )

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

        embeddings = keras.layers.Embedding(
            self.max_tokens,
            self.embedding_dimension
        )

        # Recurrent part for question body:
        rnn_body = keras.Sequential([embeddings])
        for units, reg, dropout in zip(
            self.body_layers[:-1], self.body_reg[:-1], self.body_dropout[:-1]
        ):
            rnn_body.add(
                keras.layers.Bidirectional(self.body_layer_type(
                        units,
                        activation=self.body_activation,
                        return_sequences=True,
                        kernel_regularizer=reg,
                        recurrent_regularizer=reg
                ))
            )
            if dropout:
                rnn_body.add(dropout)

        rnn_body.add(
            keras.layers.Bidirectional(self.body_layer_type(
                units=self.body_layers[-1],
                activation=self.body_activation,
                return_sequences=False,
                kernel_regularizer=self.body_reg[-1],
                recurrent_regularizer=self.body_reg[-1]
            ))
        )
        if self.body_dropout[-1]:
            rnn_body.add(self.body_dropout[-1])

        # Recurrent part for question title:
        rnn_title = keras.Sequential([embeddings])
        for units, reg, dropout in zip(
            self.title_layers[:-1],
            self.title_reg[:-1],
            self.title_dropout[:-1]
        ):
            rnn_title.add(
                keras.layers.Bidirectional(self.title_layer_type(
                        units,
                        activation=self.title_activation,
                        return_sequences=True,
                        kernel_regularizer=reg,
                        recurrent_regularizer=reg
                ))
            )
            if dropout:
                rnn_title.add(dropout)

        rnn_title.add(
            keras.layers.Bidirectional(self.title_layer_type(
                units=self.title_layers[-1],
                activation=self.title_activation,
                return_sequences=False,
                kernel_regularizer=self.title_reg[-1],
                recurrent_regularizer=self.title_reg[-1]
            ))
        )
        if self.title_dropout[-1]:
            rnn_title.add(self.title_dropout[-1])

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
            keras.Input((1,), dtype=tf.string, name='body'),
            keras.Input((1,), dtype=tf.string, name='title'),
            keras.Input((11,), dtype=tf.float32)
        ]

        # Define output:
        body_output = self.tokenizer(inputs[0])
        # body_output = keras.preprocessing.sequence.pad_sequences(
        #     body_output,
        #     maxlen=self.body_sequence_length,
        #     padding='post',
        #     truncating='post'
        # )
        body_output = body_output[:, :self.body_sequence_length]  # Truncate
        body_output = rnn_body(body_output)
        title_output = self.tokenizer(inputs[1])
        # title_output = keras.preprocessing.sequence.pad_sequences(
        #     title_output,
        #     maxlen=self.title_sequence_length,
        #     padding='post',
        #     truncating='post'
        # )
        title_output = title_output[:, :self.title_sequence_length]  # Truncate
        title_output = rnn_title(title_output)
        output = tf.concat([body_output, title_output, inputs[2]], axis=1)
        output = dnn(output)

        self.model = keras.Model(inputs, output)

        self.model.compile(
            optimizer=self.optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        print(self.model.summary())

    def fit(self):
        X = [
            self.train_data['body'].to_numpy(),
            self.train_data['title'].to_numpy(),
            self.train_data[self.numeric_features].to_numpy()
        ]

        Y = self.train_data['y'].to_numpy()

        X_val = [
            self.val_data['body'].to_numpy(),
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

        self.embeddings = self.model.layers[0].layers[0]

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
