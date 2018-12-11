# Capstone Project: Exoplanet Search

## Program: Machine Learning Engineer Nanodegree (November 2018)

## Student Name: Graciano Patino

## Installation requirements

For this project you would need Anaconda 3 with Python 3.6.

There is a requirements.txt document of the libraries needed for this project.

## For this project I used deep learning for solving the exoplanet searched project as described in Kaggle (https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/home)

The datasets for this project can be downloaded ffrom the link mentioned above.

Given required computing resources for this project, I used an instance in aws.amazon.com

The instance that I used was p2.xlarge type like the one described in the cloud computing section of the Deep Learning section of the Udacity ML Nanodegree.

## What is it in the Captone directory:

The directory contains:

1) "kepler" directory containing the datasets.
2) "requirements" directory that included the libraries required for this project.
3) "Grid Serach 1" directory contains all the files collected from first Grid Search round of simulations ran for obtaining the candidate hyperparameters.
4) "Grid Serach 2" directory contains all the files collected from second Grid Search round of simulations ran for obtaining the candidate hyperparameters.
5) There is a "get_results_v2.py" file containing functions for obtaining:
5.1) Function that plots the ROC-AUC and provides the score.
5.2) Function prints the confusion matrix considering a given threshold. This function also prints the confusion matrix.
6) exoplanet_search_data_exploration.ipynb: This file containg data exploration.
7) The directory also include 3 important simulation runs with results:
7.1) exoplanet_Nov09_8cnn_2dnn_mse_Adam1e-5_relu_30epochs.ipynb
7.2) exoplanet_Nov10_8cnn-ND_2dnn-ND_mse_Adam1e-5-betas-eps_relu_30epochs-v2.ipynb
7.3) exoplanet_Nov10_8cnn-ND_2dnn-ND_mse_Adam1e-5-betas-eps_relu_30epochs.ipynb