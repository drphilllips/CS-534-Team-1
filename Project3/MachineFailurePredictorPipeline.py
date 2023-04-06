

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def main():
    # ------
    # Step 3: Load Data and Randomly Under-Sample the Majority
    # ---
    df = pd.read_csv('ai4i2020.csv')
    print(df.columns)
    # get X and y columns
    X = df[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    y = df[['Machine failure']]
    # RandomUnderSample data to get 339 failures and 339 non-failures
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X, y)

    # ------
    # Step 4: Train-Test Split and 5-Fold Cross-Validation
    # ---
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3)
    print(y_train)


if __name__ == "__main__":
    main()
