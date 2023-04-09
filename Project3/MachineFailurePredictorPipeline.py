import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import make_classification


def main():
    # ------
    # Step 3: Load Data and Randomly Under-Sample the Majority
    # ---

    # read in csv data
    df = pd.read_csv('ai4i2020.csv')

    # get X and y columns as numpy arrays
    X = df[['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].values  # All data except label
    y = df[['Machine failure']].values.ravel()  # labels Fail/Normal

    # RandomUnderSample data to get 339 failures and 339 non-failures
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X, y)

    # ------
    # Step 4: Train-Test Split and 5-Fold Cross-Validation
    # ---

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_over, np.ravel(y_over), test_size=0.3, random_state=42)

    # define k-fold cross-validation object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ----MLP Classifier--
    mlp_hidden_layer_sizes = (100, 50)
    mlp_activation = 'relu'
    model = MLPClassifier(hidden_layer_sizes=mlp_hidden_layer_sizes, activation=mlp_activation)

    # perform 5-fold cross-validation on training data
    scores = cross_val_score(model, X, y, cv=kf)
    print(scores)

    # ----Support Vector Machine----
    svc_C = 1.0
    svc_kernel = 'rbf'
    model = SVC(C=svc_C, kernel=svc_kernel)

    # perform 5-fold cross-validation on training data
    scores = cross_val_score(model, X, y, cv=kf)
    print(scores)

    # ----Bagging Classifier----
    bgc_n_estimators = 10
    bgc_max_samples = 0.5
    model = BaggingClassifier(n_estimators=bgc_n_estimators, max_samples=bgc_max_samples)

    # perform 5-fold cross-validation on training data
    scores = cross_val_score(model, X, y, cv=kf)
    print(scores)

    # ----AdaBoost----
    adb_n_estimators = 100
    adb_learning_rate = 0.1
    model = AdaBoostClassifier(n_estimators=adb_n_estimators, learning_rate=adb_learning_rate)

    # perform 5-fold cross-validation on training data
    scores = cross_val_score(model, X, y, cv=kf)
    print(scores)

    # ----Random Forest Classifier----
    rfc_n_estimators = 100
    rfc_criterion = 'gini'
    rfc_max_features = 'sqrt'
    rfc_max_depth = None
    rfc_max_samples = 0.8
    model = RandomForestClassifier(n_estimators=rfc_n_estimators, criterion=rfc_criterion,
                                   max_features=rfc_max_features, max_depth=rfc_max_depth, max_samples=rfc_max_samples)

    # perform 5-fold cross-validation on training data
    scores = cross_val_score(model, X, y, cv=kf)
    print(scores)

    # display training data table
    trn_table = pd.DataFrame(columns=['ML Trained Model', 'Its Best Set of Parameters', 'Its F1-score on the 5-fold '
                                                                                        'Cross Validation on Training '
                                                                                        'Data (70%)'])
    trn_table.loc[0] = ['Artificial Neural Networks', '(hidden_layer_sizes, activation)', 'TBD']
    trn_table.loc[1] = ['Support Vector Machine', '(C, kernel)', 'TBD']
    trn_table.loc[2] = ['BaggingClassifier', '(n_estimators, max_samples, max_features)', 'TBD']
    trn_table.loc[3] = ['AdaBoost', '(n_estimators, learning_rate)', 'TBD']
    trn_table.loc[4] = ['Random Forest', '(n_estimators, criterion, max_depth, max_samples)', 'TBD']
    print(trn_table)

    # display testing data table
    tst_table = pd.DataFrame(columns=['ML Trained Model', 'Its Best Set of Parameters', 'Its F1-score on the testing '
                                                                                        'data(30%)'])
    tst_table.loc[0] = ['Artificial Neural Networks', '(hidden_layer_sizes, activation)', 'TBD']
    tst_table.loc[1] = ['Support Vector Machine', '(C, kernel)', 'TBD']
    tst_table.loc[2] = ['BaggingClassifier', '(n_estimators, max_samples, max_features)', 'TBD']
    tst_table.loc[3] = ['AdaBoost', '(n_estimators, learning_rate)', 'TBD']
    tst_table.loc[4] = ['Random Forest', '(n_estimators, criterion, max_depth, max_samples)', 'TBD']
    print(tst_table)


if __name__ == "__main__":
    main()
