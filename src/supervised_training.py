import argparse
import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from src.model import build_actor
from src.utils import SupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/")
    parser.add_argument('--log_path', type=str, default="logs")
    parser.add_argument('--model_path', type=str, default="models/")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    return args


def recall_m(y_true, y_pred):
    y_true, y_pred = y_true[:, 0, tf.newaxis], y_pred[:, 0, tf.newaxis]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true, y_pred = y_true[:, 0, tf.newaxis], y_pred[:, 0, tf.newaxis]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    y_true, y_pred = y_true[:, 0, tf.newaxis], y_pred[:, 0, tf.newaxis]
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == '__main__':
    # Load dataset
    args = parse_args()
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    train_dataset, steps_per_epoch = SupervisedDataset(data_path=args.data_path, train=True).get_dataset(
        batch_size=args.batch_size)
    test_dataset, validation_steps = SupervisedDataset(data_path=args.data_path, train=False).get_dataset(
        batch_size=args.batch_size)

    for (a, b), y in test_dataset.take(1):
        print(a.shape, b.shape, y.shape)
        print(a.dtype)
        break

    model = build_actor()

    # callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', factor=0.2, patience=2, min_lr=args.lr / 100),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_path, f"pretrained_sf.h5"),
            save_weights_only=True, monitor='val_f1_m',
            mode='max', verbose=True, save_best_only=True)
    ]
    # metrics
    metrics = ['acc', f1_m]
    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(),
                  metrics=metrics, )

    model.summary()
    history = model.fit(train_dataset,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=test_dataset,
                        validation_steps=validation_steps,
                        callbacks=callbacks)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(args.log_path, f"pretrained.csv"), index=False)
