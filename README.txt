# Fake News Classification Project

## Introduction

This project aims to classify news articles as real or fake using various machine learning models. The process includes exploratory data analysis, data preprocessing, and modelling using Logistic Regression, Naïve Bayes, Random Forest, and Recurrent Neural Long Short Term Memory Network (RNN LSTM).

## Files in the Repository

1. `FakeNewsClassificationEDA.ipynb`: This notebook reads from the `train.csv` file, performs data preprocessing, analyzes the data distributions, conducts hypothesis testing, performs topic modelling, and finally writes the clean preprocessed data file to `train_processed.csv`.

2. `FakeNewsClassificationModelling.ipynb`: This notebook reads from the `train_preprocessed.csv` file and performs data modelling using Logistic Regression, Naïve Bayes, and Random Forest. Once the model parameters are finalized, the models are saved as pickle files (`logreg_model.pkl`, `naivebayes_model.pkl`, `randomforest_model.pkl`). This notebook also includes model interpretability of the logistic regression model and identifies the top 20 words that have high coefficients.

3. `FakeNewsClassificationRNN.ipynb`: This notebook covers the creation, execution, and evaluation of an Recurrent Neural Long Short Term Memory Network model. It runs on Google Colab with GPU support. The notebook reads the preprocessed data from a dataframe, tokenizes the news articles using NLTK tokenizer, applies padding on the encoding of the news article, creates and fits the model on the train set, predicts the validation and test set, and generates accuracy score for validation and test predictions. The model is saved for future use in the model evaluation notebook. Note: The input file (`processed_train.csv`) path needs to be updated for running this notebook in a new environment.

4. `FakeNewsClassificationModelEvaluation.ipynb`: This notebook focuses on the evaluation of Logreg, Naïve Bayes, Random Forest and Recurrent Neural Networks classification models. It runs on Google Colab with GPU support. Performance metrics like Accuracy, Precision, Recall, F1 score and Area Under the Receiver Operating Characteristic (AUROC) are considered. The performance metrics of all models are consolidated in a Pandas DataFrame for easy comparison.

## Datasets

The datasets used in this project are `train.csv` and `train_processed.csv`.

- `train.csv`: Sourced from Kaggle (https://www.kaggle.com/competitions/fake-news/data)
- `train_processed.csv`: The preprocessed file saved in `FakeNewsClassificationEDA.ipynb`

These datasets are stored in Google Drive at https://drive.google.com/drive/folders/1gx9R5uJ3MnoQl_Oa-XJouyc416i4gWfz?usp=drive_link

## Packages

The required Python packages are listed in `requirements.txt` in pip format. Note that this does not cover the environment setup for Google Colab.

## Usage

To use the notebooks, follow the sequence:

1. Run `FakeNewsClassificationEDA.ipynb` for data preprocessing and exploratory analysis.
2. Run `FakeNewsClassificationModelling.ipynb` for modelling using Logistic Regression, Naïve Bayes, and Random Forest.
3. Run `FakeNewsClassificationRNN.ipynb` on Google Colab for RNN LSTM modelling.
4. Run `FakeNewsClassificationModelEvaluation.ipynb` for model evaluation.

Remember to adjust the data file paths according to your environment.
