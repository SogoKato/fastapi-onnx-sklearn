import json

import numpy as np
from img2feat import CNN
from skl2onnx import to_onnx
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    # Train a model.
    X, y = fetch_openml("CIFAR_10", as_frame=False, parser="liac-arff", return_X_y=True)

    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=6000, test_size=2000
    )

    X_train = reshape(X_train)
    X_test = reshape(X_test)

    clr_ = LogisticRegression(
        C=1.0, multi_class="multinomial", solver="saga", warm_start=True
    )
    clr = Classifier(
        cnn=CNN("alexnet"),
        clr=clr_,
    )
    clr.fit(X_train, y_train)

    proba, pred = clr.proba_pred(X_test)

    for i in range(len(y_test)):
        cm = "OK" if pred[i] == y_test[i] else "NG"
        print(
            "{:.4f}, {:d}, {:d}, {:s}".format(proba[i, pred[i]], pred[i], y_test[i], cm)
        )

    print("Accuracy: {:.3f}".format(acc(y_test, pred)))

    # Convert into ONNX format.
    onx = to_onnx(clr_, clr.fe(X_train[:1]))
    with open("cifar10.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # Save other data.
    dump(clr.mean, clr.scale)


def reshape(X):
    X = X.reshape(len(X), 3, 32, 32)
    X = [np.rollaxis(x, 0, 3) for x in X]
    return X


def dump(mean, scale):
    data = {
        "mean": mean.tolist(),
        "scale": float(scale),
    }
    with open("mean_scale.json", "w") as f:
        json.dump(data, f)


def acc(Y_true, Y_pred):
    return (Y_true == Y_pred).astype(np.float32).mean()


class Classifier:
    def __init__(self, cnn, clr):
        self.fe = cnn
        self.clr = clr

    def fit(self, images, y):
        X = self.fe(images)

        self.mean = X.mean(axis=0, keepdims=True)
        X = X - self.mean

        self.scale = np.sqrt(np.square(X).sum(axis=1).mean())
        X = X / self.scale

        self.clr.fit(X, y)

    def proba_pred(self, images):
        X = self.fe(images)
        X = X - self.mean
        X = X / self.scale

        proba = self.clr.predict_proba(X)

        return proba, np.argmax(proba, axis=1)


if __name__ == "__main__":
    main()
