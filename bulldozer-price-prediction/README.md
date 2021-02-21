# Predicting sale price of Bulldozers in future with machine learning

In this notebook, we're going to build a machine learning model, which is able to predict a sale price of a bulldozer based on its previous prices.

## 1. Problem definition

> How well can we predict a bulldozer price in future, based on our previous sold's data (bulldozers data have sold)?

## 2. Data

The data is downloaded from Kaggle: https://www.kaggle.com/c/bluebook-for-bulldozers/data

We have 3 base dataset:

* Train.csv is the training set, which contains data through the end of 2011.
* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
* Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

## 3. Evaluation

The evaluation metric for this competition is the `RMSLE` (root mean squared log error) between the actual and predicted auction prices.

For more information: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

**Note:** The goal of most regression evaluation metrics is to minimise the error. For example, our goal for this project will be building a model which minimise RMSLE.

## 4. Features

All of features have described in data dictionary that provided by Kaggle:
https://docs.google.com/spreadsheets/d/1zM62fC9pCteimqcUvGzAoJfo50MDrszD9vyyXta2nYg/edit?usp=sharing