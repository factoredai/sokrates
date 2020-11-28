from .Model import Model
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file):
    df = pd.read_csv(file)

    def class_decider(row):
        if row['score'] < 2.:
            return 0
        elif row['score'] > 2.:
            return 1
        else:
            if row['n_answers'] < 1.:
                return 0
            else:
                return 1

    df.loc[:, 'y'] = df.apply(class_decider, axis=1)

    data = df[[
        'body',
        'title',
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
        'title_question_marks',
        'y'
    ]]

    train_data, val_data = train_test_split(
        data,
        test_size=0.3,
        stratify=df['y'],
        random_state=42
    )

    return train_data, val_data


def experiment(file, specs, tokenizer=None, normalizer=None):
    '''
    Inputs:

        file: path to a fileto which results can be appended

        specs: expermient specifications

        tokenizer: keras ``TextVectorization`` layer adapted tp
                   training data (optional)

        normalizer: keras normalization layer adapted to
                    training data (optional)


    Outputs:

        model: trained keras model

        results: dictionary with specifications and validation accuracy


    Effects:

        - appends results to the json file specified in ``file``
    '''

    specs = specs.copy()
    if tokenizer:
        specs['tokenizer'] = tokenizer
    if normalizer:
        specs['normalizer'] = normalizer

    assert 'data_path' in specs

    if 'train_data' not in specs or 'val_data' not in specs:
        specs['train_data'], specs['val_data'] = load_data(specs['data_path'])

    model = Model(specs)
    model.fit()

    results = {
        k: v for k, v in specs.items() if k not in {
            'train_data', 'val_data', 'tokenizer', 'normalizer'
        }
    }

    results['train_accuracy'] = model.history['train_accuracy'][-1]
    results['val_accuracy'] = model.history['val_accuracy'][-1]

    with open(file, 'a') as fopen:
        json.dump(results, fopen)
        fopen.write('\n')

    return model, results
