import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier


def main():
    # ------
    # Step 3: Load Data and Randomly Under-Sample the Majority
    # ---
    df = pd.read_csv('ai4i2020.csv')
    # get X and y columns
    X = df[['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']] # All data except label
    y = df[['Machine failure']] # labels Fail/Normal
    # RandomUnderSample data to get 339 failures and 339 non-failures
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_over, y_over = undersample.fit_resample(X, y)
    # ------
    # Step 4: Train-Test Split and 5-Fold Cross-Validation
    # ---
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_over, np.ravel(y_over), test_size=0.3, random_state=42)

    # MLP Classifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("MLP Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)

    # SV Classifier
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print("SV Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)

    X_class, y_class = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    clf = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0).fit(X_class, y_class)
    clf.predict([[0, 0, 0, 0]])

    ml_table = pd.DataFrame(columns=['ML Trained Model', 'Its Best Set of Parameters', 'Its F1-score on the 5-fold '
                                                                                       'Cross Validation on Training '
                                                                                       'Data (70%)'])
    print(ml_table)


if __name__ == "__main__":
    main()
