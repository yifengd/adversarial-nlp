import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


def main(params: dict):
    X = np.load(params.shap_files_path + "shap_vals.npy")
    Y = np.load(params.shap_files_path + "shap_labels.npy")
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=0, shuffle=True, train_size=0.8
    )
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    preds = gnb.predict(x_test)
    # draw_confusion_matrix(y_test, preds, "Gaussian Naive Bayes")
    print(classification_report(y_test, preds))
