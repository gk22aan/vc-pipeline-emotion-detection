import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        print(f"Error: The file {params_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {params_path}: {e}")
    except KeyError as e:
        print(f"Error: Missing key in params file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading parameters: {e}")

def read_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty.")
    except pd.errors.ParserError:
        print(f"Error: Parsing error while reading {csv_path}.")
    except Exception as e:
        print(f"An unexpected error occurred while reading data: {e}")

def process_data(df):
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]
        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing column in DataFrame: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing data: {e}")

def save_data(data_path, train_data, test_data):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except FileExistsError:
        print(f"Error: The directory {data_path} already exists.")
    except IOError as e:
        print(f"Error: I/O error while saving data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving data: {e}")

def main():
    try:
        test_size = load_params('params.yaml')
        if test_size is None:
            print("Error: Failed to load test size from parameters.")
            return
        
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        if df is None:
            print("Error: Failed to read data.")
            return
        
        final_df = process_data(df)
        if final_df is None:
            print("Error: Failed to process data.")
            return
        
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        print("Data processing and saving completed successfully.")
    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
