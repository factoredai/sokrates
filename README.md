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
- The `app` directory contains all code necessary to run the HTTP service.
- Most code is contained in the `app/app_core` package.
- The `app_core.data_processing` package contains data extraction and preprocessing
  functionalities. Within it:

  - The `text_extract` package contains classes used to extract features from 
    text. They should all follow the `Extractor` interface.

  - The `XMLparser` module contains functionality to parse the StackExchange `.xml`
    files and convert them to dataframes.

  - The `make_dataset_csv` module uses `text_extract` and `XMLparser` to process the
    StackExchange `.xml` files and export them as csv.

- The `app_core.ml_models` package contains managers (wrappers) to handle the ML models
  themselves.

- The `basic_nlp_model` package contains code to quickly build and test neural network
  models on the dataset.

- The `notebooks` directory contains several Jupyter notebooks with data exploration
  and model testing.

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
You can also add an optional third argument (`True` or `False`) to force the
re-processing of existing csvs.

### Running Simple Baseline
To run the first simple baseline of the model run:
```bash
python -m ml_models
```
This will then prompt you for the title of your question and the body,
which could be a path to a file where the question is stored as rendered
HTML.

### Running the Server (Docker)
To start the HTTP server with docker, do the following:
- First, [install Docker](https://www.docker.com/).
- Second, prepare your `.env` file. It must contain the variables specified
  in the `app/.env.example` file. Note that you must have access to the S3
  bucker where we are storing our models!
- Third, navigate to the `app` directory and build the docker image with:
```shell script
docker build -t sokrates:<version> .
```
- Finally, run the container with
```shell script
docker run -p 3000:3000 --env-file <path-to-.env-file> sokrates:<version>
```


[Back to top](#sokrates)
