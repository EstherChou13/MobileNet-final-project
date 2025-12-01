import argparse
import tensorflow as tf
from mobile_net import MobileNet
from tensorflow.keras.datasets import cifar10

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, x_test, y_test

def train(alpha):
    x_train, y_train, x_test, y_test = load_data()
    model = MobileNet(alpha=alpha)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, 
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=64
    )

    print("Training finished for alpha =", alpha)
    model.save(f"experiments/mobilenet_alpha_{alpha}.keras")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    train(args.alpha)

if __name__ == "__main__":
    main()
