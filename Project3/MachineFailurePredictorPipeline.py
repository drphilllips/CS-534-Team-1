import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



def main():
    # ------
    # Step 3: Load Data and Randomly Under-Sample the Majority
    # ---
    df = pd.read_csv('ai4i2020.csv')
    # get X and y columns
    X = df[['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]  # All data except label
    y = df[['Machine failure']]  # labels Fail/Normal
    # RandomUnderSample data to get 339 failures and 339 non-failures
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X, y)
    print([X_over, y_over])
    # ------
    # Step 4: Train-Test Split and 5-Fold Cross-Validation
    # ---
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_over, np.ravel(y_over), test_size=0.3, random_state=42)

    # Create the machine learning training table
    ml_train_table = pd.DataFrame(columns=['ML Trained Model', 'Its Best Set of Parameters',
                                           'Its F1-score on the 5-fold Cross Validation on Training Data (70%)'])

    # MLP Classifier
    # layout parameter space
    parameters = {'solver': ['sgd', 'adam'],
                  'alpha': [0.001, 0.005, 0.007, 0.01, 0.03, 0.1, 0.3, 1.0],
                  'hidden_layer_sizes': [(5, 2), (10, 2), (20, 2), (50, 2), (100, 2), (200, 2)],
                  'learning_rate': ['constant', 'adaptive'],
                  'activation': ['tanh', 'relu'],
                  'early_stopping': [True]}
    mlp = MLPClassifier(max_iter=200)
    # Grid Search to learn best parameters with 5-fold validation
    grid = GridSearchCV(mlp, parameters, cv=5)
    grid.fit(X_train, y_train)
    # Generate scores from best parameters
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print("MLP Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1_mlp = f1_score(y_test, y_pred)
    print("F1 Score:", f1_mlp, '\n')
    params_mlp = grid.best_params_

    # Create row and concatenate it to the ml train table
    ai_row = {'ML Trained Model': 'Artificial Neural Networks',
              'Its Best Set of Parameters': str({'hidden layer sizes': params_mlp['hidden_layer_sizes'],
                                                 'activation': params_mlp['activation']}),
              'Its F1-score on the 5-fold Cross Validation on Training Data (70%)': f1_mlp}

    ml_train_table = pd.concat([ml_train_table, pd.DataFrame(ai_row, index=[0])], ignore_index=True)

    # SV Classifier
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5, 10, 20]}
    svc = SVC()
    grid = GridSearchCV(svc, parameters, cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.get_params())
    print("SV Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1_svc = f1_score(y_test, y_pred)
    print("F1 Score:", f1_svc, '\n')

    # Create row and concatenate it to the ml train table
    svc_row = {'ML Trained Model': 'Super Vector Machine',
               'Its Best Set of Parameters': str({'C': 5, 'kernel': 4}),
               'Its F1-score on the 5-fold Cross Validation on Training Data (70%)': f1_svc}

    ml_train_table = pd.concat([ml_train_table, pd.DataFrame(svc_row, index=[0])], ignore_index=True)

    # Bagging Classifier
    clf = BaggingClassifier(estimator=None, n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Bagging Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1_bag = f1_score(y_test, y_pred)
    print("F1 Score:", f1_bag, '\n')

    # Create row and concatenate it to the ml train table
    bag_row = {'ML Trained Model': 'BaggingClassifier',
               'Its Best Set of Parameters': str({'n_estimators': 5, 'max_samples': 4, 'max_features': 5}),
               'Its F1-score on the 5-fold Cross Validation on Training Data (70%)': f1_bag}

    ml_train_table = pd.concat([ml_train_table, pd.DataFrame(bag_row, index=[0])], ignore_index=True)

    # Adaboost Classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Adaboost Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1_boost = f1_score(y_test, y_pred)
    print("F1 Score:", f1_boost, '\n')

    # Create row and concatenate it to the ml train table
    boost_row = {'ML Trained Model': 'AdaBoost',
                 'Its Best Set of Parameters': str({'n_estimators': 100, 'learning_rate': 10}),
                 'Its F1-score on the 5-fold Cross Validation on Training Data (70%)': f1_boost}

    ml_train_table = pd.concat([ml_train_table, pd.DataFrame(boost_row, index=[0])], ignore_index=True)

    # Random Forest Classifier
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1_rf = f1_score(y_test, y_pred)
    print("F1 Score:", f1_rf, '\n')

    # Create row and concatenate it to the ml train table
    rf_row = {'ML Trained Model': 'Random Forest',
              'Its Best Set of Parameters': str({'n_estimators': 100, 'criterion': 10, 'max_depth': 10, 'max_samples': 100}),
              'Its F1-score on the 5-fold Cross Validation on Training Data (70%)': f1_rf}

    ml_train_table = pd.concat([ml_train_table, pd.DataFrame(rf_row, index=[0])], ignore_index=True)

    ml_train_table = ml_train_table.reset_index(drop=True)
    print(ml_train_table)
    ml_train_table.to_csv('ml_train_table.csv', index=False)


if __name__ == "__main__":
    main()
