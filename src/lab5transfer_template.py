import datetime
import itertools
import os
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm
from tqdm.keras import TqdmCallback


def get_convert_datasets(batch_size=32):
    # label_names = {
    #     0: 'airplane',
    #     1: 'automobile',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer',
    #     5: 'dog',
    #     6: 'frog',
    #     7: 'horse',
    #     8: 'ship',
    #     9: 'truck',
    # }
    cat_indices = [3]
    (x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

    # Normalize pixel values to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # convert the ray labels of 10 classes to our binary cat or not cat labels
    y_train_cats = None # FIXME: as in lab 4 convert to binary labels
    y_test_cats = None# FIXME: as in lab 4 convert to binary labels

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cats))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_cats))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def val_split_dataset(dataset, validation_split=0.2):
    # Calculate the number of samples
    total_size = sum(1 for _ in dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # Split the dataset
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    return train_ds, val_ds    


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fig_folder='./figures'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    abs_fig_folder = os.path.abspath(os.path.join(os.getcwd(),fig_folder))
    if not os.path.isdir(abs_fig_folder):
        os.makedirs(abs_fig_folder)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(abs_fig_folder, str(title)+'.png'))


def visualize_model(model: Model,
                    x_visualize: np.ndarray,
                    y_visualize: np.ndarray,
                    model_name='Model1',
                    fig_folder='./figures'):
    """
    Visualize our predictions using classification report and confusion matrix

    :param model: the model used to make predictions for visualization
    :param x_visualize: the input features given used to generate prediction
    :param y_visualize: the true output to compare against the predictions
    """
    y_pred = model.predict(x_visualize)
    y_pred = np.array(y_pred > 0.5, dtype=int)
    y_true = y_visualize.reshape(-1,1)
    class_names = ['not cat', 'cat']

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title=f'{model_name} Confusion matrix, without normalization',
                          fig_folder=fig_folder)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title=f'{model_name} Normalized confusion matrix',
                          fig_folder=fig_folder)


def visualize_model_dataset(model: Model,
                            dataset: tf.data.Dataset,
                            eval_samples=6000,
                            model_name='Model1',
                            fig_folder='./figures'):
    """
    Uses a model and a dataset to visualize the current predictions

    :param model: the model used to make predictions
    :param dataset: the dataset to use for generating predictions and comparing against truth
    :param eval_samples: the number of samples used to evaluate from the dataset
    """
    x_visualize = []
    y_visualize = []

    total_samples = 0
    img_batch: np.ndarray
    label_batch: np.ndarray
    for img_batch, label_batch in dataset.as_numpy_iterator():
        total_samples += img_batch.shape[0]
        x_visualize.append(img_batch)
        y_visualize.append(label_batch[:, None])
        if total_samples > eval_samples:
            break

    x_visualize = np.vstack(x_visualize)
    y_visualize = np.vstack(y_visualize)

    visualize_model(model, x_visualize, y_visualize, model_name=model_name, fig_folder=fig_folder)


def main():
    base_dir = Path('/opt', 'data', 'cifar10-tfrecords')
    fig_folder = './figures'

    force_make_tfrecords = False
    validation_split = 0.3
    num_shuffle_batches = 3
    visualize_dataset = False #NOTE: this will not work on ACEHUB. Optional.

    data_img_shape = (32, 32, 3)

    train_model = True
    train_pre_trained_model = True
    batch_size = 32
    epochs = 10

    show_metrics_for_valid_dataset = True

    # get the datasets
    train_dataset, test_dataset = get_convert_datasets(batch_size)

    # # set aside some training files for validation
    train_dataset, valid_dataset = val_split_dataset(train_dataset, validation_split=validation_split)

    # check our dataset by visualizing it. Note this can go forever if you let it
    # NOTE: this does not work on acehub
    if visualize_dataset:
        for img_batch, label_batch in train_dataset.as_numpy_iterator():
            plt.imshow(img_batch[0])
            print(label_batch[0])
            plt.show()

    # the cifar dataset comes with these fixed values so hard code them
    total_train_samples = 5000 * 10
    total_train_cats = 5000 * 1
    total_train_not_cats = 5000 * 9

    total_test_samples = 1000 * 10
    total_test_cats = 1000 * 1
    total_test_not_cats = 1000 * 9

    print(
        f"Total Samples: {total_train_samples}\nTotal Cats   : {total_train_cats}\nAccuracy if I always guess not cat: {(total_train_samples - total_train_cats) / total_train_samples * 100}%")

    # make our class weighting to balance our training since there are many more not cats
    class_weight = {
        0: None, # FIXME: 
        1: None, # FIXME:
    }

    saved_model_filename = 'model.h5'
    if not os.path.exists(saved_model_filename) or train_model:
        input_tensor = Input(shape=data_img_shape)

        #TODO: replace this section with lab4 network

        output_tensor = None # FIXME:

        model = Model(input_tensor, output_tensor)

        #TODO: train lab4 model on TFRecords

        model.save(saved_model_filename)
    else:
        model = load_model(saved_model_filename)

    eval_dataset = valid_dataset if show_metrics_for_valid_dataset else test_dataset
    visualize_model_dataset(model=model,
                            dataset=eval_dataset,
                            eval_samples=total_test_samples)

    # Now for a pre-trained network
    # This is similar to the example in Section 5.3 from the Chollet book
    saved_model_filename = 'model_pretrained.h5'
    if not os.path.exists(saved_model_filename) or train_pre_trained_model:
        conv_base = None # FIXME: load Resnet50 here
        # TODO: freeze the base layers to not be trainable
    
        print("Conv Base Summary")
        conv_base.summary()

        # TODO: add layers on top to make a binary classifier
        input_tensor = None # FIXME

        output_tensor = None # FIXME

        model_pretrained = Model(input_tensor, output_tensor)

        #TODO: Compile the new model

        print("Full Model Summary")
        model_pretrained.summary()

        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=1),
                     TqdmCallback(epochs=epochs, data_size=total_train_samples, batch_size=batch_size)]

        # TODO: Train the model where resnet50 pieces are frozen

        # Get results and CM to see how it's going
        print('RESULTS FOR FIRST HALF OF PRETRAINED')
        eval_dataset = valid_dataset if show_metrics_for_valid_dataset else test_dataset
        visualize_model_dataset(model=model_pretrained,
                                dataset=eval_dataset,
                                eval_samples=total_test_samples,
                                model_name='Transfer1 ',
                                fig_folder=fig_folder)

        # unfreeze the whole thing and do more training, but with less epochs than before
        # TODO: unfreeze the conv_base

        callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=1),
                     TqdmCallback(epochs=epochs, data_size=total_train_samples, batch_size=batch_size)]

        # TODO: Train the model more, but with less epochs this time

        model_pretrained.save(saved_model_filename)
    else:
        model_pretrained = load_model(saved_model_filename)

    # Get results for second phase of learning
    print('RESULTS FOR SECOND STAGE PRETRAINED')
    eval_dataset = valid_dataset if show_metrics_for_valid_dataset else test_dataset
    visualize_model_dataset(model=model_pretrained,
                            dataset=eval_dataset,
                            eval_samples=total_test_samples,
                            model_name='Transfer2 ',
                            fig_folder=fig_folder)


if __name__ == "__main__":
    main()
