# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Import numpy for numerical operations
import numpy as np

# Import Pandas for data manipulation 
import pandas as pd

# Import os for operating system-related functions
import os

# Import the 'train_test_split' function to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split 

# Import data scalers for preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

# Ignore warnings to suppress non-essential messages
import warnings
warnings.filterwarnings("ignore")

# Import matplotlib and seaborn for data visualization
import matplotlib.pyplot as plt
import seaborn as sns



# +
############################## AQUIRE & PREPARE wine-quality FUNCTIONS ##############################

def wrangle_wine():
    """
    This function,'wrangle_wine' brings in two csv files, replaces spaces with underscores, 
    changes all letters to lowercase, creates a categroical column on red wine,
    and concatenates them into one pandas dataframe. It then renames two columns,
    and changes the null values from our categorical column to 0 where it is white wine and 1 when it is red.
    
    Returns:
    train (DataFrame): Training data.
    validate (DataFrame): Validation data.
    test (DataFrame): Test data.
    """
    #if os.path.isfile('wine.csv'):
   # else:
        #creates new csv if one does not already exist
        #print('Download the .csv from https://data.world/food/wine-quality')
    white = pd.read_csv('winequality_white.csv')
    red = pd.read_csv('winequality_red.csv')
    white.columns = white.columns.str.replace(' ', '_')
    white.columns = white.columns.str.lower()
    red.columns = red.columns.str.replace(' ', '_')
    red.columns = red.columns.str.lower()
    red['is_red'] = 1
    data = [white, red]
    df = pd.concat(data)
    red['is_red'] = 1
    # Rename columns
    df = df.rename(columns={
    'free_sulfur_dioxide': 'free_so2',
    'total_sulfur_dioxide': 'total_so2'
    })
    df.is_red = df.is_red.fillna(0).astype(int)
    df['high_quality'] = df.quality.map({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1})
    # Write that dataframe to disk for later. Called "caching" the data for later.
    df.to_csv('wine.csv')
    train, validate, test = train_val_test(df)
    return train, validate, test
    
def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    info
    shape
    outliers
    description
    missing data stats
    
    Args:
    df (DataFrame): The DataFrame to be summarized.
    k (float): The threshold for identifying outliers.
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    print('Descriptions:')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('======================\n======================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('int64').describe().T.to_markdown())
    print('======================\n======================')
    print('missing values:')
    print('by column:')
    print(missing_by_col(df).to_markdown())
    print('by row: ')
    print(missing_by_row(df).to_markdown())
    print('======================\n======================')
    print('Outliers: ')
    print(report_outliers(df, k=k))
    print('======================\n======================')
def null_counter(df):
    """
    Count the number and percentage of missing values in each column of a DataFrame.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    DataFrame: A DataFrame containing columns 'name', 'num_rows_missing', and 'pct_rows_missing'.
    """
    new_columns = ['name', 'num_rows_missing', 'pct_rows_missing']
    new_df = pd.DataFrame(columns=new_columns)
    for col in list(df.columns):
        num_missing = df[col].isna().sum()
        pct_missing = num_missing / df.shape[0]
        add_df = pd.DataFrame([{'name': col, 'num_rows_missing': num_missing,
                               'pct_rows_missing': pct_missing}])
        new_df = pd.concat([new_df, add_df], axis=0)
    new_df.set_index('name', inplace=True)
    return new_df
def null_dropper(df, prop_required_column, prop_required_row):
    """
    Remove columns and rows with missing values based on specified proportions.

    Args:
    df (DataFrame): The DataFrame to remove missing values from.
    prop_required_column (float): Proportion of non-missing values required for columns.
    prop_required_row (float): Proportion of non-missing values required for rows.

    Returns:
    DataFrame: The DataFrame with missing values removed.
    """
    prop_null_column = 1 - prop_required_column
    for col in list(df.columns):
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
    row_threshold = int(prop_required_row * df.shape[1])
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    return df
def train_val_test(df):
    """
    Split a DataFrame into training, validation, and test sets.

    Args:
    df (DataFrame): The DataFrame to split.

    Returns:
    train (DataFrame): Training data.
    validate (DataFrame): Validation data.
    test (DataFrame): Test data.
    """
    train_val, test = train_test_split(df,
                                  random_state=1349,
                                  train_size=0.8)
    train, validate = train_test_split(train_val,
                                  random_state=1349,
                                  train_size=0.7)
    return train, validate, test
def scale_data(train,
               validate,
               test,
               columns_to_scale=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_so2', 'total_so2', 'density', 'ph', 'sulphates', 'alcohol'],
               return_scaler=False):
    '''
    Scales the 3 data splits.
    Takes in train, validate, and test data splits and returns their scaled counterparts using Min-Max scaling.
    
    Args:
    train (DataFrame): Training data.
    validate (DataFrame): Validation data.
    test (DataFrame): Test data.
    columns_to_scale (list): List of column names to scale.
    return_scaler (bool): If True, return the scaler object.

    Returns:
    If return_scaler is True:
        scaler (MinMaxScaler): Scaler object.
        train_scaled (DataFrame): Scaled training data.
        validate_scaled (DataFrame): Scaled validation data.
        test_scaled (DataFrame): Scaled test data.
    If return_scaler is False:
        train_scaled (DataFrame): Scaled training data.
        validate_scaled (DataFrame): Scaled validation data.
        test_scaled (DataFrame): Scaled test data.

    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
def missing_by_col(df):
    """
    Count the number of missing values by column in a DataFrame.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    Series: A Series with column names as the index and the count of missing values as values.
    """
    return df.isnull().sum(axis=0)
def missing_by_row(df):
    """
    Generate a report on the number and percentage of rows with a certain number of missing columns.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    DataFrame: A DataFrame with columns 'num_cols_missing', 'percent_cols_missing', and 'num_rows'.
    """
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df
def get_fences(df, col, k=1.5) -> tuple:
    """
    Calculate upper and lower fences for identifying outliers in a numeric column.

    Args:
    df (DataFrame): The DataFrame containing the column.
    col (str): The name of the column to analyze.
    k (float): The threshold multiplier for the IQR.

    Returns:
    float: Lower bound for outliers.
    float: Upper bound for outliers.
    """
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound
def report_outliers(df, k=1.5) -> None:
    """
    Report outliers in numeric columns of a DataFrame based on the specified threshold.

    Args:
    df (DataFrame): The DataFrame to analyze.
    k (float): The threshold for identifying outliers.

    Returns:
    None
    """
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')
def get_continuous_feats(df) -> list:
    """
    Find continuous numerical features in a DataFrame.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    list: List of column names containing continuous numerical features.
    """
    num_cols = []
    num_df = df.select_dtypes('number')
    for col in num_df:
        if num_df[col].nunique() > 20:
            num_cols.append(col)
    return num_cols
def split_data(df, target=None) -> tuple:
    """
    Split a DataFrame into training, validation, and test sets with optional stratification.

    Args:
    df (DataFrame): The DataFrame to split.
    target (Series): Optional target variable for stratified splitting.

    Returns:
    train (DataFrame): Training data.
    validate (DataFrame): Validation data.
    test (DataFrame): Test data.
    """
    train_val, test = train_test_split(
        df,
        train_size=0.8,
        random_state=1349,
        stratify=target)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1349,
        stratify=target)
    return train, validate, test
def display_numeric_column_histograms(data_frame):
    """
    Display histograms for numeric columns in a DataFrame with three colors.

    Args:
    data_frame (DataFrame): The DataFrame to visualize.

    Returns:
    None(prints to console)
    """
    numeric_columns = data_frame.select_dtypes(exclude=["object", "category"]).columns.to_list()
    # Define any number of colors for the histogram bars
    colors = ["#FFBF00"]
    for i, column in enumerate(numeric_columns):
        # Create a histogram for each numeric column with two colors
        figure, axis = plt.subplots(figsize=(10, 3))
        sns.histplot(data_frame, x=column, ax=axis, color=colors[i % len(colors)])
        axis.set_title(f"Histogram of {column}")
        plt.show()

# -


