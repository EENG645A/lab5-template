'''
EENG-645 Lab 5: Scaling up with Ray
Reference: https://docs.ray.io/en/latest/tune/getting-started.html
'''

# Import the necessary modules
import os
import sklearn.metrics
import itertools
import numpy as np
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray import train
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler #these are aliased in tune (same thing)
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune.search.optuna import OptunaSearch

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, load_model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fig_folder='./figures'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # abs_fig_folder = os.path.abspath(os.path.join(os.getcwd(),fig_folder))
    if not os.path.isdir(fig_folder):
        os.makedirs(fig_folder)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
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
    file_path = os.path.join(fig_folder, str(title)+'.png')
    print(file_path)
    plt.savefig(file_path)


def visualize_model(model: Model,
                    x_visualize: np.ndarray,
                    y_visualize: np.ndarray,
                    class_names=[],
                    model_name='Model1',
                    fig_folder='./figures'):
    """
    Visualize our predictions using classification report and confusion matrix
    This is tailored to the full cifar10 dataset using the original 'y' or 'labels'

    :param model: the model used to make predictions for visualization
    :param x_visualize: the input features given used to generate prediction
    :param y_visualize: the true output to compare against the predictions
    """
    y_pred = model.predict(x_visualize)
    y_pred = y_pred.argmax(axis=1)
    y_true = y_visualize
    

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title=f'{model_name} Confusion matrix, without normalization',
                          fig_folder=fig_folder)

    # Plot normalized confusion matrix
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title=f'{model_name} Normalized confusion matrix',
                          fig_folder=fig_folder)

ray.shutdown() # Kill Ray incase it didn't stop cleanly in another run

# Define folder paths
# NOTE: no need to start/use TensorBoard, ray already does that in here
ray_results = '/remote_home/Lab5/Lab5-template/ray_results'
# fig_path = '/remote_home/Lab5/Lab5-template/Figures'
# model_path = '/remote_home/Lab5/Lab5-template/models'

# Define resources to use per sample. 
# Ray will see total available CPU/GPU available and divide these into that,
# Running that many trials in parallel
NUM_CPU = 2
NUM_GPU = 1 
num_samples = 10
local_mode = False


def create_model(config):
    '''config: a dictionary of hyperparameters'''
    model = None #FIXME: Make the CNN network here and return it as model
    return model


# Define the training function
def train_model(config):
    '''This is your "trainable" that ray will run each sample trial'''
    # Load the CIFAR10 dataset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # TODO: Normalize the data, 
    
    # TODO: change y_train/y_test categorical

    
    # Create the model
    model = create_model(config)

    # Train the model
    history = model.fit(x_train, 
                y_train, 
                batch_size=config["batch_size"], 
                epochs=config["epochs"], 
                validation_split=0.2,
                verbose=0, # this is running inside ray workers so will spam terminal if not 0
                # required to report each epoch to ray instead of each training run. Also checkpoints each training run at interval chosen
                callbacks=[ReportCheckpointCallback(
                  checkpoint_on="train_end", # save model at end, not every epoch
                  report_metrics_on='epoch_end', # make training_iteration==epoch
                )] 
    )

    # Evaluate the model
    # NOTE: in a real experiment this "test" is like a second layer of validation post trial
    # we would still want another test split that we evaluate after all the hyperparam tuning
    # but we won't do that here to keep things short. This will show in Tensorboard
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]

    # # Feel free to uncomment this if you want to see what's in these variables
    # print('score variable is', score)
    # print('history', history)
    # print('history.history', history.history)

    # Report the accuracy to Tune
    # this will make the accuracy reported at the end of training runs this evaluate accuracy
    train.report({
        'eval_accuracy': accuracy,
        'accuracy': history.history['accuracy'][-1], # accuracies at train *end* (last epoch)
        'val_accuracy': history.history['val_accuracy'][-1]
    })

# Initialize Ray
ray.init(local_mode=local_mode) # set true for better debugging, but need be false for scaling up

# Define the search space--each "sample" run will sample from each of these
# TODO: use tune.loguniform and tune.choice to pick hyperparameters
# for each tune.choice choose 2-3 sensible values
search_space = {
    "lr": None, # FIXME:
    "filters": None, # FIXME:
    "hidden": None, # FIXME:
    "batch_size": None, # FIXME:
    "epochs": None, # FIXME:
}

###########################################################
# Version 1 start - For this lab, use a 'scheduler' which is how ray searches
# There are several schedulers, but ASHA is simple and effective
# Run only this block (version 1) or the next block (version 2)--not both
###########################################################

# # TODO: Create an Asynchronous HyperBand scheduler (ASHA)
# # https://docs.ray.io/en/latest/tune/api/schedulers.html
scheduler = None # FIXME

# # The newer way of running tune is to make a 'Tuner' object then call .fit() on it
tuner = tune.Tuner(
    tune.with_resources(train_model, resources={"cpu": NUM_CPU, "gpu": NUM_GPU}),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=train.RunConfig(
        name="lab5ASHA",
        # stop={"accuracy": 0.99},
        verbose=3,
        local_dir=ray_results, # run tensorboard in this dir later
    ),
    param_space=search_space, #NOTE: This is the what will be passed to "train_model" func above
)
results = tuner.fit()

###########################################################
# Version 1 end
###########################################################


###########################################################
# Version 2 start - Using a 'search_alg' instead of a scheduler
# Ray can act as a wrapper to call another hyperparam search library like 
# optuna, hyperopt, sklearn, etc. This is done in place of a "scheduler"
###########################################################

# TODO: Now comment out the block "Verion 1" above, and instead create an OptunaSearch
# and repeat the same using a search_alg instead of a scheduler

###########################################################
# Version 2 end
###########################################################
df = results.get_dataframe()
print(df)

best_config = results.get_best_result(metric="val_accuracy", mode="max")
print(f"Best result: {best_config}")

# Note where the checkpoint for the best result is and load the model
best_checkpoint_folder = best_config.checkpoint.path
best_model_path = os.path.join(best_checkpoint_folder, 'model.keras')
best_model = load_model(best_model_path)
print('The BEST model EVER summary: ')
best_model.summary()

# # Get test data again
_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = (x_test - 128) / 128.8
# y_test = tf.keras.utils.to_categorical(y_test, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
visualize_model(best_model,
                x_visualize=x_test,
                y_visualize=y_test,
                class_names=class_names,
                model_name='Best Model',
                fig_folder='./figures')

# This gives you hyperparam insight, not Confusion Matrix, etc.
# After analyzing the correlations between hyperparams,
# you could use this information to train what you think is the "best"
# model and repeat further analysis as in lab 4 (but you are not required
# to do that in this lab--you may stop here.)
# Ray was set to checkpoint after each trial, so all 10 sample models 
# could be retrieved later by loading the checkpoint in Ray.