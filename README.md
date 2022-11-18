# Data Science Project: FAANG Forecaster

## Project Description
The goal of the project is to forecast the closing stock prices of companies such as Facebook (Meta), Amazon, Apple, Netflix and Google in the next seven days using previous three weeks' of historical closing prices.

## Data Collection
I used **Yahoo Finance API** to extract the historical stock prices of FAANG based on daily frequency. Since it is a multilabel regression problem I had to frame it to a traditional machine learning framework in order to be able to use the algorithms. I used **sliding window** technique to create features and for each label there are 21 features. The parameters are following.

- **Horizon** - The number of values to be forecasted into the future. As I wanted the models to be able to forecast the next seven days, the horizon was **seven**.
- **Window Size** - The number of features used to forecast the horizon. As I wanted to use the three weeks' of historical closing stock prices, the window size is **21**.

It is common to use either three, two or one week's of historical data for a stock price forecasting problem because increasing the window size will actually make it difficult for a model to learn patterns as the main focus should always be on building a model that learns the patterns from the most recent history.

## Modeling
I used most of the traditional machine learning algorithms which you can find in the following table.

| Algorithms|Used|
| -------- | -------- |
|Bayesian Ridge|True|
|Linear Regression|True|
|Support Vector Machine|True|
|K Nearest Neighbors|True|
|Decision Tree|True|
|Random Forest|True|
|Adaptive Boosting (AdaBoost)|True|
|Light Gradient Boosted Machine (LightGBM)|True|
|Gradient Boosting (GBM)|True|
|Extreme Gradient Boosting (XGBoost)|True|
|Category Boosting (CatBoost)|True|

In most cases, Bayesian Ridge, Linear Regression and Support Vector Machine prevailed. The main evaluation metric for this project is **Mean Absolute Error (MAE)**.

|Model |Algorithms|MAE|
| -------- | -------- | -------- |
| Facebook|Bayesian Ridge|9.59|
| Amazon |Linear Regression|3.53|
| Apple |Linear Regression|1.82|
| Netflix |Bayesian Ridge|15.36|
| Google |Linear Regression|2.46|

The **average loss** of the FAANG Forecaster is **6.55**. The image below displays the performance of the model on **Facebook** stock prices.

![](https://i.imgur.com/3LrQIUl.jpg)

The image below displays the performance of the model on **Amazon** stock prices.

![](https://i.imgur.com/yZTqI68.jpg)

The image below displays the performance of the model on **Apple** stock prices.

![](https://i.imgur.com/KPnJffY.jpg)

The image below displays the performance of the model on **Netflix** stock prices.

![](https://i.imgur.com/wJiGXuF.jpg)

The image below displays the performance of the model on **Google** stock prices.

![](https://i.imgur.com/sibvUdX.jpg)
