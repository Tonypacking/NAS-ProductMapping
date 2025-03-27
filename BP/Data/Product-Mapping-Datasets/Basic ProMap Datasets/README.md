# Basic ProMap Datasets
This part of the repository contains newly created and released datasets for product mapping: ProMapCz and ProMapEn. 
Both datasets contain manually created product pairs from two e-shops: Alza-Mall for ProMapCz and Walmart-Amazon for ProMapEn. We extracted several pieces of information about each product: name, specification, short description, long descriptions, images and price. Moreover, each product pair contains also the type of match and product category.


It also contains existing preprocessed datasets used for comparison with our ProMap datasets originally from [Amazon-Google](https://dbs.uni-leipzig.de/en) and [Amazon-Walmart](https://hpi.de/naumann/projects/repea\-ta\-bility/datasets/amazon-walmart-dataset.html) datasets.

Each folder contains 4 datasets:
* *dataset_name*-**train_data.csv**, which contains original train data
* *dataset_name*-**test_data.csv**, which contains original test data
* *dataset_name*-**train_data_similaities.csv**, which contains precomputed similarities from train data
* *dataset_name*-**test_data_similarities.csv**, which contains precomputed similarities from test data

## ProMapCz
Contains 1495 product pairs for product mapping task in Czech language of which 1196 pairs are for training and 299 are for testing data. 

Datasets:
* promap_cz-test_data.csv
* promap_cz-train_data.csv
* promap_cz-train_data_similarities.csv
* promap_cz-test_data_similarities.csv


## ProMapEn
Contains 1555 product pairs for product mapping task in English language of which 1244 pairs are for training and 311 are for testing data. 

Datasets:
* promap_en-train_data.csv
* promap_en-test_data.csv
* promap_en-train_data_similarities.csv
* promap_en-test_data_similarities.csv


## Amazon-Google
Contains 3235 product pairs for product mapping task in English language of which 2588 pairs are for training and 647 are for testing data. 

Datasets:
* amazon_google-train_data.csv
* amazon_google-test_data.csv
* amazon_google-train_data_similarities.csv
* amazon_google-test_data_similarities.csv

## Amazon-Walmart
Contains 3143 product pairs for product mapping task in English language of which 2514 pairs are for training and 629 are for testing data. 

Datasets:
* amazon_walmart-train_data.csv
* amazon_walmart-test_data.csv
* amazon_walmart-train_data_similarities.csv
* amazon_walmart-test_data_similarities.csv
