# Sokrates

## Contents

* [About](#about)
* [Repository Contents](#repository-contents)
* [Instructions and Usage](#instructions-and-usage)
  * [Setup](#setup)
  * [Building the Dataset](#building-the-dataset)
  * [Running Simple Baseline](#running-simple-baseline)

## About
'Sokrates' is an ML-powered assistant to help write better questions!

## Repository Contents
- The `data_processing` package contains data extraction and preprocessing
  functionalities. Within it:

  - The `text_extract` package contains classes used to extract features from 
    text. They should all follow the `Extractor` interface.

  - The `XMLparser` module contains functionality to parse the StackExchange `.xml`
    files and convert them to dataframes.

  - The `make_dataset_csv` module uses `text_extract` and `XMLparser` to process the
    StackExchange `.xml` files and export them as csv.

- The `ml_models` package contains managers (wrappers) to handle the ML models
  themselves.

- The `basic_nlp_model` package contains code to quickly build and test neural network
  models on the dataset.

- The `Exploration.ipynb` notebook contains exploratory analyses of variables as well
  as some basic model tests.

## Instructions and Usage

### Setup
In order to run the project you must first ensure that the required packages
are installed, which you can do with:
```bash
pip install -r requirements.txt
```
You must also install the nltk dependencies. To do this, run the following in a
`python` session:
```python
import nltk
nltk.download("punkt")  # Punkt for tokenizing
```

### Building the dataset
To build the dataset, first you must have downloaded and decompressed the data files
from the stack exchange data dump. After this you will have a collection of directories
(one per topic) containing `.xml` files. If `mydir` is the directory that contains these
folders and `outdir` is the directory where you want to store the output csvs, you can 
generate them with:
```bash
python -m data_processing mydir outdir
```

### Running Simple Baseline
To run the first simple baseline of the model run:
```bash
python -m ml_models
```
This will then prompt you for the title of your question and the body,
which could be a path to a file where the question is stored as rendered
HTML.


[Back to top](#sokrates)
