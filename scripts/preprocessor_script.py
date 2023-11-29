import pandas as pd
import click
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn import set_config


@click.command
@click.option('--raw_train_data', type=str, help="Path for accessing raw train data")
@click.option('--raw_test_data', type=str, help="Path for accessing raw test data")
# @click.option('--output_dropped_train', type=str, help="Path for writing file containing train data with dropped features")
# @click.option('--output_dropped_test', type=str, help="Path for writing file containing test data with dropped features")
@click.option('--output_preprocessed_data', type=str, help="Path for writing files containing preprocessed data")
@click.option('--output_preprocessor', type=str, help="Path for writing the file containing the preprocessor object")
@click.option('--seed', type=int, help="Random seed", default=522)

def main(raw_train_data, raw_test_data, output_preprocessed_data, output_preprocessor, seed):
    """
    This script drops columns which have too many null values to be useful for analysis as well as NA values,
    then splits the dataframe into the train and test splits and saves them to csv files, 
    with 10% of the data in the train split to speed up the training time.
    The data is then preprocessed for use in model training, and the preprocessed data
    is also stored in csv files.
    """
    # Read train and test data 
    bank_marketing_train = pd.read_csv(raw_train_data)
    bank_marketing_test = pd.read_csv(raw_test_data)

    # Hardcoding uninsightful columns to drop (based on TA advice)
    bank_marketing_train = bank_marketing_train.drop(["poutcome", "contact"], axis = 1)
    bank_marketing_test = bank_marketing_test.drop(["poutcome", "contact"], axis = 1)
    # Drop NAs
    bank_marketing_train = bank_marketing_train.dropna()
    bank_marketing_test = bank_marketing_test.dropna()
    # Splitting to X and y test and train
    X_train = bank_marketing_train.drop(columns=subscribed)
    y_train = bank_marketing_train[subscribed]
    X_test = bank_marketing_test.drop(columns=subscribed)
    y_test = bank_marketing_test[subscribed]

    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Making column transformer
    numeric_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital']
    ordinal_features = ['education']
    binary_features = ['default', 'housing', 'loan']
    drop_features = ['day_of_week', 'month']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    education_order = ['primary', 'secondary', 'tertiary']
    ordinal_transformer = OrdinalEncoder(categories=[education_order], dtype=int)
    binary_transformer = OneHotEncoder(drop = 'if_binary', dtype=int, handle_unknown = "ignore", sparse_output=False)

    preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (ordinal_transformer, ordinal_features),
    (binary_transformer, binary_features),  
    (categorical_transformer, categorical_features),
    ("drop", drop_features),  
    )
    
    # Pickle unfit preprocessor object
    pickle.dump(preprocessor, open(os.path.join(output_preprocessor, "preprocessor.pickle"), "wb"))

    # Fit preprocessor
    preprocessor.fit(X_train)
    preprocessed_X_train = preprocessor.transform(X_train)
    preprocessed_X_test = preprocessor.transform(X_test)

    # Write raw data "data/processed" directory
    preprocessed_X_train.to_csv(os.path.join(output_preprocessed_data, "preprocessed_X_train.csv"))
    preprocessed_X_test.to_csv(os.path.join(output_preprocessed_data, "preprocessed_X_test.csv"))


if __name__ == "__main__":
    main()