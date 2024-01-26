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


# the following snippet is from https://www.tensorflow.org/tutorials/load_data/tfrecord
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# this function was added since the _int64_feature takes a single int and not a list of ints
def _int64_list_feature(value: typing.List[int]):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# Function for reading images from disk and writing them along with the class-labels to a TFRecord file.
def convert(x_data: np.ndarray,
            y_data: np.ndarray,
            out_path: Path,
            records_per_file: int = 4500,
            ) -> typing.List[Path]:
    """
    Function for reading images from disk and writing them along with the class-labels to a TFRecord file.

    :param x_data: the input, feature data to write to disk
    :param y_data: the output, label, truth data to write to disk
    :param out_path: File-path for the TFRecords output file.
    :param records_per_file: the number of records to use for each file
    :return: the list of tfrecord files created
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Open a TFRecordWriter for the output-file.
    record_files = []
    n_samples = x_data.shape[0]
    # Iterate over all the image-paths and class-labels.
    n_tfrecord_files = int(np.ceil(n_samples / records_per_file))
    for idx in tqdm(range(n_tfrecord_files),
                    desc="Convert Batch",
                    total=n_tfrecord_files,
                    position=0):
        record_file = out_path / f'train{idx}.tfrecord'
        record_files.append(record_file)
        slicer = slice(idx * records_per_file, (idx + 1) * records_per_file)
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for x_sample, y_sample in tqdm(zip(x_data[slicer], y_data[slicer]),
                                           desc="Convert Image in batch",
                                           total=records_per_file,
                                           position=1,
                                           leave=False):
                # Convert the ndarray of the image to raw bytes. note this is bytes encodes as uint8 types
                img_bytes = x_sample.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = {
                    'image': _bytes_feature(img_bytes),
                    'label': _int64_list_feature(y_sample)
                }
                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)
                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)
                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
    return record_files


def convert_datasets(base_dir: Path):
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

    # make folders for train and test
    base_train_dir = base_dir / 'train'
    base_test_dir = base_dir / 'test'

    # convert the ray labels of 10 classes to our binary cat or not cat labels
    y_train_cats = None # FIXME: same as lab 4
    y_test_cats = None # FIXME: same as lab 4

    # convert the numpy arrays to tfrecords
    train_files = convert(x_data=x_train, y_data=y_train_cats, out_path=base_train_dir)
    test_files = convert(x_data=x_test, y_data=y_test_cats, out_path=base_test_dir)

    return train_files, test_files


def get_dataset(filenames: typing.List[Path], img_shape: tuple) -> tf.data.Dataset:
    """
    This function takes the filenames of tfrecords to process into a dataset object
    The _parse_function takes a serialized sample pulled from the tfrecord file and
    parses it into a sample with x (input) and y (output) data, thus a full sample for training

    This function will not do any scaling, batching, shuffling, or repeating of the dataset

    :param filenames: the file names of each tf record to process
    :param img_shape: the size of the images a width, height, channels
    :return: the dataset object made from the tfrecord files and parsed to return samples
    """

    def _parse_function(serialized):
        """
        This function parses a serialized object into tensor objects to use for training
        NOTE: you must use tensorflow functions in this section
        using non-tensorflow function will not get the results you expect and/or hinder performance
        see how each function starts with `tf.` meaning the function is form the tensorflow library
        """
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)
        # convert the image shape to a tensorflow object
        image_shape = tf.stack(img_shape)

        # get the raw feature bytes
        image_raw = parsed_example['image']
        # Decode the raw bytes so it becomes a tensor with type.
        image_inside = tf.io.decode_raw(image_raw, tf.uint8)
        # cast to float32 for GPU operations
        image_inside = tf.cast(image_inside, tf.float32)
        # reshape to correct image shape
        image_inside = tf.reshape(image_inside, image_shape)

        # get the label and convert it to a float32
        label = tf.cast(parsed_example['label'], tf.float32)

        # return a single tuple of the (features, label)
        return image_inside, label

    # the tf functions takes string names not path objects, so we have to convert that here
    filenames_str = [str(filename) for filename in filenames]
    # make a dataset from slices of our file names
    files_dataset = tf.data.Dataset.from_tensor_slices(filenames_str)

    # make an interleaved reader for the TFRecordDataset files
    # this will give us a stream of the serialized data interleaving from each file
    dataset = files_dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x),
                                       # 12 was picked for the cycle length because there were 12 total files,
                                       # and I wanted to cycle through all of them
                                       cycle_length=12,  # how many files to cycle through at once
                                       block_length=1,  # how many samples from each file to get
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       deterministic=False)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(map_func=_parse_function,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


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
    y_true = y_visualize
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

    # get tfrecord file names (and create if necessary
    if force_make_tfrecords or not os.path.exists(base_dir):
        train_files, test_files = convert_datasets(base_dir)
    else:
        train_files = [base_dir / 'train' / file_name for file_name in os.listdir(base_dir / 'train')]
        test_files = [base_dir / 'test' / file_name for file_name in os.listdir(base_dir / 'test')]

    # set aside some training files for validation
    valid_files = train_files[-int(len(train_files) * validation_split):]

    # make the tf datasets from the files
    train_dataset: tf.data.Dataset = get_dataset(train_files, img_shape=data_img_shape)
    valid_dataset: tf.data.Dataset = get_dataset(valid_files, img_shape=data_img_shape)
    test_dataset: tf.data.Dataset = get_dataset(test_files, img_shape=data_img_shape)

    # function to do the scaling on our images (this is just another possible way of doing it)
    def mapper(image_inside_batched, label):
        image_inside_batched = image_inside_batched / 128.  # scale from 0 to 2
        image_inside_batched = image_inside_batched - 1  # Zero-center
        return image_inside_batched, label

    # repeat, shuffle, scale, batch, and prefetch the datasets in a function
    def repeat_shuffle_scale_batch_prefetch_dataset(dataset: tf.data.Dataset):
        # repeat
        dataset = dataset.repeat()
        # shuffle
        dataset = dataset.shuffle(buffer_size=batch_size * num_shuffle_batches)
        # map
        dataset = dataset.map(map_func=mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # chain together batch and prefetch
        dataset = dataset.batch(batch_size=batch_size).prefetch(1)
        return dataset

    # process our datasets in a pipeline
    train_dataset = repeat_shuffle_scale_batch_prefetch_dataset(train_dataset)
    valid_dataset = repeat_shuffle_scale_batch_prefetch_dataset(valid_dataset)
    test_dataset = repeat_shuffle_scale_batch_prefetch_dataset(test_dataset)

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
