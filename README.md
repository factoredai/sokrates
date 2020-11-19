# Sokrates

## Contents

* [About](#about)
* [Repository Contents](#repository-contents)
* [Instructions and Usage](#instructions-and-usage)
  * [Setup](#setup)

## About
'Sokrates' is an ML-powered assistant to help write better questions!

## Repository Contents

- The `text_extract` package contains classes used to extract features from 
  text. They should all follow the `Extractor` interface.

- The `XMLparser` module contains functionality to parse the StackExchange `.xml`
  files and convert them to dataframes.

- The `make_dataset_csv` module uses `text_extract` and `XMLparser` to process the
  StackExchange `.xml` files and export them as csv.

## Instructions and Usage

### Setup
In order to run the project you must first ensure that the required packages
are installed, which you can do with:
```bash
pip install -r requirements.txt
```
You must also install the nltk dependencies. To do this, do the following in a
`python` session:
```python
import nltk
nltk.download("punkt")
```

### Building the dataset


[Back to top](#sokrates)
