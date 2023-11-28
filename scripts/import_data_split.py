import click
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--data_id', type=int, help="Dataset ID")
@click.option('--ratio', type=float, help="Train, test split")

def main(data_id, ratio):
    """Imports data and splits it"""
    bank_marketing = fetch_ucirepo(id=data_id) 
  
    # bank marketing data
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    bank_marketing_data = pd.concat([X, y], axis=1)

    bank_marketing_data.rename(columns={'y': 'subscribed'}, inplace=True)

    # create a preliminary split to explore data
    bank_marketing_train, bank_marketing_test = train_test_split(
        bank_marketing_data, train_size=ratio, stratify=bank_marketing_data["subscribed"]
    )

    # write raw data "data/raw" directory
    bank_marketing_train.to_csv("data/raw/bank_marketing_train.csv")
    bank_marketing_test.to_csv("data/raw/bank_marketing_test.csv")

if __name__ == '__main__':
    main()