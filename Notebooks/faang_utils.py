# Importing Dependencies
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler, PowerTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import set_config

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import skopt
import yaml
import os

# Setting a font scale, grid, and palette for a plot
sns.set(font_scale = 1.5, style = "darkgrid", palette = "bright")

# Filtering warnings
warnings.filterwarnings(action = "ignore")

# Configuring a setting to display all the columns
pd.options.display.max_columns = None

# Configuring a setting to visualize a pipeline
set_config(display = "diagram")

# Defining a global level seed
np.random.seed(seed = 42)

# Loading the configuration file
with open(file = "../Configuration/config.yml") as yaml_file:
    config = yaml.safe_load(stream = yaml_file)

# Loading the window size
WINDOW_SIZE = config.get('parameters').get('window_size')

# Loading the forecasting horizon
HORIZON = config.get('parameters').get('horizon')

# Loading the evaluation metric
TARGET_PATH = config.get("target_path")

# Loading the evaluation metric
METRIC = config.get("metric")

# Loading the Facebook (Meta) model
with open(file = "../Models/meta_ml_forecaster.pickle", mode = "rb") as pickled_file:
    meta_model = pickle.load(file = pickled_file)
    
# Loading the Amazon model
with open(file = "../Models/amazon_ml_forecaster.pickle", mode = "rb") as pickled_file:
    amazon_model = pickle.load(file = pickled_file)
    
# Loading the Apple model
with open(file = "../Models/apple_ml_forecaster.pickle", mode = "rb") as pickled_file:
    apple_model = pickle.load(file = pickled_file)
    
# Loading the Netflix model
with open(file = "../Models/netflix_ml_forecaster.pickle", mode = "rb") as pickled_file:
    netflix_model = pickle.load(file = pickled_file)
    
# Loading the Google model
with open(file = "../Models/google_ml_forecaster.pickle", mode = "rb") as pickled_file:
    google_model = pickle.load(file = pickled_file)

# Defining a function to load historical data
def load_data(ipo_name = None,
              target_path = TARGET_PATH):
    """
    This is a function to load historical data.
    
    Args:
        ipo_name: A name of an initial public offering.
        target_path: A path to a target folder.
        
    Returns:
        A Pandas data frame.
    """
    # Assigning the extension to an ipo
    filename = f"{ipo_name.lower()}.parquet.brotli"
    
    # Creating a filepath
    filepath = os.path.join(target_path, filename)
    
    # Loading the dataset
    data_frame = pd.read_parquet(path = filepath, engine = "fastparquet")
    
    # Removing the potential spaces and lowering the column names
    data_frame.columns = data_frame.columns.str.strip().str.lower()
    
    # Renaming the close variable as target
    data_frame.rename(columns = {"close":"target"}, inplace = True)
    
    # Casting the data type of date variable from object to datetime
    data_frame["date"] = pd.to_datetime(arg = data_frame["date"], yearfirst = True)
    
    # Sorting the observations based on date in ascending order 
    data_frame = data_frame.sort_values(by = "date").reset_index(drop = True)
    
    # Selecting date and target variables
    data_frame = data_frame[["date", "target"]]
    
    # Returning the data frame
    return data_frame

# Defining a function to create features and labels using sliding window techinque
def create_features_labels(data_frame = None, 
                           window_size = WINDOW_SIZE, 
                           horizon = HORIZON):
    """
    This function is used to create a data frame using sliding window technique.
    
    Args:
        data_frame: A Pandas data frame.
        window_size: A window size to contain features.
        horizon: A forecasting horizon.
        
    Returns:
        A Pandas data frame.
    """
    # Creating empty lists for features and labels
    features = []
    labels = []
    
    # Looping through each observation in data frame
    for i in range(data_frame.shape[0]):
        # Creating a window
        window = data_frame.target[i:window_size + i].tolist()
        
        # Creating a forecasting horizon
        target = data_frame.target[window_size + i:window_size + i + horizon].tolist()
        
        # Creating a condition based on the length
        if len(window) == window_size and len(target) == horizon:
            # Appending the windows to the list
            features.append(window)
            
            # Creating a condition based on forecasting horizon
            if horizon > 1:
                # Appending a target to the list
                labels.append(target)
            else:
                # Appending a target to the list
                labels.append(target[0])
            
            # Creating an assertion
            assert len(window) == window_size
            assert len(target) == horizon
        else:
            # Passing in case the condition is not satisfied 
            pass
    
    # Converting features and labels into arrays
    features = np.array(object = features)
    labels = np.array(object = labels)
    
    # Returning the features and labels
    return features, labels

# Defining a function to create forecasting pipeline
def build_pipeline(regressor = None,
                   apply_feature_scaling = False,
                   scaler_type = None,
                   support_multioutput = True,
                   apply_bayesian_optimization = False,
                   hyperparameters = None,
                   train_features = None,
                   train_labels = None,
                   n_iterations = 50,
                   metric = METRIC,
                   verbosity = 0):
    """
    This function is used to evaluate the performance of a forecaster.
    
    Args:
        regressor: A regressor instance.
        apply_feature_scaling: Whether or not to apply feature scaling.
        scaler_type: A type of feature scaler.
        support_multioutput: Whether or not a regressor supports multiouput regression.
        train_features: An array or data frame containing independent variables for train set.
        apply_bayesian_optimization: Whether or not to apply hyperparameter tuning using bayesian optimization.
        train_features: Training features as an array.
        train_labels:  Training labels as an array.
        n_iterations: A number of iterations of iterations for hyperparameter.
        metric: An evaluation metric to optimize a model.
        verbosity: A level of verbosity to display an output.
        
    Returns:
        A Pandas data frame.
    """
    # Creating a dictionary of evaluation metrics
    metrics_dict = {"r_squared":"r2",
                    "mae":"neg_mean_absolute_error",
                    "mse":"neg_mean_squared_error",
                    "rmse":"neg_root_mean_squared_error",
                    "mape":"neg_mean_absolute_percentage_error"}
    
    # Creating a dictionary of features scalers
    scalers_dict = {"robust": RobustScaler(),
                    "minmax": MinMaxScaler(),
                    "maxabs": MaxAbsScaler(),
                    "standard": StandardScaler(),
                    "power": PowerTransformer(method = "box-cox")}
    
    # Creating a condition based on feature scaling and multioutput regressor
    if not apply_feature_scaling and support_multioutput:
        pipe = Pipeline(steps = [("imputer", SimpleImputer(strategy = "median")),
                                 ("regressor", regressor)])
    elif not apply_feature_scaling and not support_multioutput:
        pipe = Pipeline(steps = [("imputer", SimpleImputer(strategy = "median")),
                                 ("regressor", MultiOutputRegressor(estimator = regressor, n_jobs = -1))])
    elif apply_feature_scaling and not support_multioutput:
        pipe = Pipeline(steps = [("imputer", SimpleImputer(strategy = "median")),
                                 ("feature_scaler", scalers_dict.get(scaler_type)),
                                 ("regressor", MultiOutputRegressor(estimator = regressor, n_jobs = -1))])
    else:
        pipe = Pipeline(steps = [("imputer", SimpleImputer(strategy = "median")),
                                 ("feature_scaler", scalers_dict.get(scaler_type)),
                                 ("regressor", regressor)])
    
    # Creating a condition to apply hyperparameter tuning
    if not apply_bayesian_optimization:
        # Fitting the train features and labels
        pipe.fit(X = train_features, y = train_labels)
    else:
        # Defining an operating level seed
        np.random.seed(seed = 42)
        
        # Instantiating the cross validation technique
        tscv = TimeSeriesSplit()
        
        # Instatiating the hyperparameter tuning method
        bayes_search_cv = skopt.BayesSearchCV(estimator = pipe, 
                                              search_spaces = hyperparameters, 
                                              n_iter = n_iterations, 
                                              scoring = metrics_dict.get(metric), 
                                              n_jobs = -1, 
                                              cv = tscv, 
                                              verbose = verbosity,
                                              random_state = 42)

        # Fitting the train features and labels
        bayes_search_cv.fit(X = train_features, y = train_labels)

        # Extracting the best pipeline
        pipe = bayes_search_cv.best_estimator_
    
    # Returning the regressor pipeline
    return pipe

# Defining a function to visually compare the forecasts to ground truths
def plot_forecasts(model = None, 
                   train_features = None, 
                   train_labels = None, 
                   test_features = None, 
                   test_labels = None):
    """
    This function is used to visually compare forecasts to ground truth
    
    Args:
        model: A regressor instance.
        data_frame: A Pandas data frame.
        train_features: An array or data frame containing independent variables for train set.
        train_labels: A dependent variable for train set.
        test_features: An array or data frame containing independent variables for test set.
        test_labels: A dependent variable for test set.
        
    Returns:
        Matplotlib graphs.
    """
    # Flattening the train & test labels
    train_labels = train_labels.ravel()
    test_labels = test_labels.ravel()
    
    # Forecasting on train and test sets
    train_forecasts = model.predict(X = train_features).ravel()
    test_forecasts = model.predict(X = test_features).ravel()
    
    # Horizontally stacking the forecasts & ground truths
    stacked_forecasts = np.hstack(tup = (train_forecasts, test_forecasts))
    stacked_labels = np.hstack(tup = (train_labels, test_labels))
    
    # Visually comparing the forecasts to ground truths 
    plt.figure(figsize = (20, 30))
    plt.subplot(3, 1, 1)
    plt.plot(range(stacked_labels.size), stacked_labels, label = "Ground Truth", c = "teal")
    plt.plot(range(stacked_forecasts.size), stacked_forecasts, label = "Forecasts", c = "red")
    plt.title(label = "Ground Truth vs Forecasts", fontsize = 16)
    plt.ylabel(ylabel = "Stock Price ($)", fontsize = 16)
    plt.xlabel(xlabel = "Label Index", fontsize = 16)
    plt.legend(loc = "best", fontsize = 16)

    plt.subplot(3, 1, 2)
    plt.plot(range(train_labels.size), train_labels, label = "Train Ground Truth", c = "teal")
    plt.plot(range(train_forecasts.size), train_forecasts, label = "Train Forecasts", c = "red")
    plt.title(label = "Train Set Stock Price ($)", fontsize = 16)
    plt.ylabel(ylabel = "Stock Price ($)", fontsize = 16)
    plt.xlabel(xlabel = "Label Index", fontsize = 16)
    plt.legend(loc = "best", fontsize = 16)

    plt.subplot(3, 1, 3)
    plt.plot(range(test_labels.size), test_labels, label = "Test Ground Truth", c = "teal")
    plt.plot(range(test_forecasts.size), test_forecasts, label = "Test Forecasts", c = "red")
    plt.title(label = "Test Set Stock Price ($)", fontsize = 16)
    plt.ylabel(ylabel = "Stock Price ($)", fontsize = 16)
    plt.xlabel(xlabel = "Label Index", fontsize = 16)
    plt.legend(loc = "best", fontsize = 16)
    plt.show()
    
# Defining a function to evaluate the performance of a model
def evaluate_model_performance(model = None,
                               train_features = None,
                               train_labels = None,
                               test_features = None,
                               test_labels = None,
                               algorithm_name = None,
                               metric = METRIC):
    """
    This function is used to evaluate the performance of a forecaster.
    
    Args:
        model: A regressor instance.
        train_features: An array or data frame containing independent variables for train set.
        train_labels: A dependent variable for train set.
        test_features: An array or data frame containing independent variables for test set.
        test_labels: A dependent variable for test set.
        algorithm_name: A name of an algorithm used to build a regressor pipeline.
        
    Returns:
        A Pandas data frame.
    """
    # Creating a dictionary of evaluation metrics
    metrics_dict = {"r_squared":"r2",
                    "mae":"neg_mean_absolute_error",
                    "mse":"neg_mean_squared_error",
                    "rmse":"neg_root_mean_squared_error",
                    "mape":"neg_mean_absolute_percentage_error"}
    
    # Making predictions on train & test set
    train_forecasts = model.predict(X = train_features)
    test_forecasts = model.predict(X = test_features)
    
    # Evaluating the performance of a model on train set using evaluation metrics for a forecasting problem
    mape_train = round(number = sum(abs((train_labels.ravel() - train_forecasts.ravel()) / train_labels.ravel())) / train_labels.ravel().size, ndigits = 2)
    mse_train = round(number = sum(pow(base = train_labels.ravel() - train_forecasts.ravel(), exp = 2)) / train_labels.ravel().size, ndigits = 2)
    mae_train =  round(number = sum(abs(train_labels.ravel() - train_forecasts.ravel())) / train_labels.ravel().size, ndigits = 2)
    rmse_train = round(number = np.sqrt(mse_train), ndigits = 2)
    
    # Evaluating the performance of a model on test set using evaluation metrics for a forecasting problem
    mape_test = round(number = sum(abs((test_labels.ravel() - test_forecasts.ravel()) / test_labels.ravel())) / test_labels.ravel().size, ndigits = 2)
    mse_test = round(number = sum(pow(base = test_labels.ravel() - test_forecasts.ravel(), exp = 2)) / test_labels.ravel().size, ndigits = 2)
    mae_test =  round(number = sum(abs(test_labels.ravel() - test_forecasts.ravel())) / test_labels.ravel().size, ndigits = 2)
    rmse_test = round(number = np.sqrt(mse_test), ndigits = 2)
    
    # Creating a dictionary of the evaluation metrics
    metrics_dict = {"MAPE Train": mape_train,
                    "MAPE Test": mape_test,
                    "MAE Train": mae_train,
                    "MAE Test": mae_test,
                    "MSE Train": mse_train,
                    "MSE Test": mse_test,
                    "RMSE Train": rmse_train,
                    "RMSE Test": rmse_test}
    
    # Storing the evaluation metrics in data frame
    evaluation_df = pd.DataFrame(data = metrics_dict, index = [algorithm_name])
    
    # Returning the data frame
    return evaluation_df

# Defining a function to forecast the the closing stock price of the next seven days
def forecast_closing_stock_prices(ipo_name = None):
    # Creating a dictionary of models
    models_dict = {"Meta": meta_model,
                   "Amazon": amazon_model,
                   "Apple": apple_model,
                   "Netflix": netflix_model,
                   "Google": google_model}
    
    # Loading the historical data of the Initial Public Offering (IPO)
    data_frame = load_data(ipo_name = ipo_name)
    
    # Creating an array of input features
    input_data = data_frame.iloc[-(WINDOW_SIZE + HORIZON):-HORIZON, 1].to_numpy()
    
    # Extracting the ground truth
    ground_truth = data_frame.iloc[-HORIZON:, 1].to_numpy()
    
    # Extracting the dates
    dates = data_frame.iloc[-HORIZON:, 0].dt.date.tolist()
    
    # Expanding the dimension of input features from 1D to 2D
    input_data = np.expand_dims(a = input_data, axis = 0)
    
    # Extracting the forecasting model from the dictionary
    model = models_dict.get(ipo_name.capitalize())
    
    # Forecasting the closing stock prices of the next seven days
    forecasts = model.predict(X = input_data).ravel()
    
    # Creating data dictionary
    data_dict = {"Date": dates,
                 "Actual": ground_truth, 
                 "Forecasted": forecasts}
    
    # Storing dates, actual and forecasted stock prices in a data frame
    results_df = pd.DataFrame(data = data_dict)
    
    # Setting the Date column as the index
    results_df.set_index(keys = "Date", inplace = True)
    
    # Visually comparing the forecasts to actual stock prices
    results_df.plot(kind = "bar", color = ["teal", "red"])
    plt.title(label = f"{ipo_name.capitalize()} Closing Stock Prices ($)", fontsize = 16)
    plt.ylabel(ylabel = "Prices ($)", fontsize = 16)
    plt.xlabel(xlabel = "Dates", fontsize = 16)
    plt.legend(bbox_to_anchor = [1, 1])
    plt.xticks(rotation = 0)
    plt.show()