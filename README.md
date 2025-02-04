# Lab 5

*This lab has a new requirements.txt file (in .devcontainer folder) that includes libraries not installed in previous labs. You may use a new container image for your acehub environment to have these preinstalled:* `git.antcenter.net:4567/nyielding/acehub-tensorflow-image:1.3-ray` or do `pip install -Ur ./.devcontainer/requirements.txt`

*You may build/use this container on your own machine by using "reopen in devcontainer" from the repo in VSCode. There are comments in the `./.devcontainer/devcontainer.json` file* 

In this lab we will not quite follow the 7-step process because we will be building upon Lab 4 from last week. If your Lab 4 did not work, or you are unsure, ask for help to make sure your base solution works or else you will be fighting problems from two labs at once.

Remember our problem is to identify cat pictures from the CIFAR dataset. We will still have to separate the cats from the other pictures as we did before--this applies to the **transfer learning** problem in this lab. For the problem using Ray for a hyperparam search, we will use the *full 10-class cifar10 dataset* to avoid over complicating the assignment.

## Lab Files:
`lab5transfer.py`: (student work) Transfer learning concept 

`lab5ray.py`: (student work) Scaling up with Ray--hyperparam searching 

`heatmap.ipynb`: (code provided) Demonstration of visualizing CNN insight

<!-- ## TensorFlow Records OUTDATED, REMOVED TF RECORD STUDENT WORK WI25 QUARTER
This lab give example code for using TensorFlow Records (TFRecords) in the *transfer learning* problem. While we will use the exact same data from the previous lab (the CIFAR-10 dataset) we will assume for some reason we need to read it from disk. The most likely reason for this is that the whole dataset cannot fit in RAM at once. However, we may be able to read our data in small chunks which we convert to TFRecord files. TFRecords allow us to read small chunks of TFRecord files dynamically while training and avoid reading the entire dataset into RAM at once. 

Students will take the CIFAR-10 dataset and process it into TFRecords. When writing the TFRecords, only pre-process the labels. Do *NOT* scale the input data before writing the TFRecords. There should be some split of the records so that there are different dataset objects for training, validation, and testing. It should only be necessary to have about 10-20 TFRecords in total since our dataset is actually pretty small.

Remember you will need to parse the TFRecords into your desired tensor shapes (like for images) and then repeat, shuffle, scale, batch, and prefetch them as necessary for training. *Code for accomplishing this is provided*. The student must use the TFRecords as the data for training in the *transfer learning* portion. -->

## Lab 4 Network
Train your final network from Lab 4 as a point of comparison. Verify your performance is about the same as before. 

## Pre-Trained Network

We will use a pre-trained network instead of training from scratch as in Lab 4. Load the Keras built in pre-trained CNNs *ResNet50* to modify for our cat detector problem. This should involve removing the old output layer and adding a new one to match the output of our current problem.

### Real Life Note
I know Keras has CNNs with weights trained for CIFAR-10, so you might be tempted to use the original output of the CNN. However, we are in a learning setting, so we will pretend we cannot do that and must strip off the old output layer and retrain them at least a little. *Instead, use the 'imagenet' weights for the pretrained network*

### CNN Modification
Take your pretrained CNN and retrain the weights for our problem. This will involve freezing most of the layers close to the input and re-training or maybe even adding new layers to train. I expect that the performance should be better compared to our from-scratch network but this isn't necessary. However, you should at least get similar performance.  

You can decide if you just want to add more layers or retrain existing layers in the model.

Print out the performance of this pre-trained ANN on the CIFAR-10 test set using a TFRecords dataset. Make sure you use all the samples from the test set even though you are generating them with a TFRecords dataset.  Remember that you should only print the test set when you believe the network is fully trained and will generalize.

## Visualize Insight for CNNs
This section of the lab is a modified example from the Chollet book. The example can be found [here,](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/660498db01c0ad1368b9570568d5df473b9dc8dd/first_edition/5.4-visualizing-what-convnets-learn.ipynb). This technique uses Class Activation Maps to highlight in the original image the sections the CNN keyed in on to make a classification. Think of this as answering the question "why did you pick that class?". This is a very useful technique for understanding the inner workings of CNNs.

In this section we will explore some visualizations for CNNs. Since everyone has different models that may or may not have trained well, everyone will use a pre-trained model from keras. The template code contains the startup code for the `VGG16` model trained on ImageNet. 

Running the notebook will perform a heat map analysis like the Elephant picture from the Chollet book example. Using the cat/dog images provided make a heat map on what parts of the image the CNN think is cat or like.

The images used are found in the `img_in` folder. Any images that fit a label from imagenet can be placed in here to try yourself. 

The notebook will output the top 3 predicted classes from imagenet and a plot in `img_out` that overlays the heatmap of the CNN layer activation. 

### Help
Note the example from Chollet is with an older version of TensorFlow that did not have eager execution. In order to run the example (and thus use example code in your project) you must include the line:
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```
to disable eager execution and be able to use the gradient commands the way the example does.

## Hyperparameter Searching with Ray
During your labs, when seeking to improve performance you likely found yourself changing parameters such as layer sizes, learning rate, etc. and then running the training over, looking at the results, then repeating. The 7 step process discussed in this course is certainly important and should continued to be used for finding a good model that performs well without overfitting. When trying to take the extra step from a 'good' model to a finely tuned model, the number of tweakable parameters can become overwhelming. This is where tools such as Ray Tune can help.

**NOTE**: To avoid complications with the imbalanced dataset, the full cifar10 dataset will be used for this portion of the lab. This could be solved with using custom metrics instead of 'accuracy' but is beyond the scope here.

### Resources
For guidance on how to use Ray Tune, start with these two links:

`https://docs.ray.io/en/latest/tune/getting-started.html`

`https://docs.ray.io/en/latest/tune/key-concepts.html`

**NOTE:** Ray Tune is built to work with both Tensorflow and PyTorch. For this class you must use Tensorflow. Some of the examples in the Ray docs use PyTorch, so be careful to not blindly copy/paste code. There are several APIs for using Ray libraries and this can be confusing--the library has a huge scope, is fast moving, and frequently depreciates code. There may even be lines of code in examples on the website that are depreciated and thus no longer work as expected. 

Be sure to read the doc for any function you use to make sure it is not depreciated, and avoid mixing API styles together. Most of the code has been provided in the lab, particularly the pieces that could most likely trip you up. The library may seem to have a steep learning curve, but once you understand the concepts it is fairly simple to apply.

### Tuning instructions
Follow along the TODOs and FIXMEs in the code, filling in the blanks. The intent is to create a 'trainable' that loads the cifar10 dataset and creates a CNN model similar to lab4 but with classification of all 10 classes and parameters that are sampled for tuning.

Define your search space by sampling the parameters from a choice of 2-3 sensible options, or a loguniform or uniform distribution as appropriate for the parameter.

Once you have defined your trainable and parameter search space, you will perform **TWO** hyperparam search, labeled as "Version 1" and "Version 2" in the code template comments. Both of these searches should be conducted with **10 samples** and expected to take 15-20min runtime depending on what parameters were sampled.

*Version 1*: perform a search using Ray directly with the `scheduler` ASHA (AsynchronousHyperband).

*Version 2*: perform a search using Ray as a wrapper for the `search_alg` Optuna.

For grading on this portion, copy/paste the final "Trial Status" table from the terminal at the end of training as a comment at the end of the code. Do this for both Version 1 and 2. Include the figures for confusion matrices of the best model for each one of the "versions" as an image in your repo.

**NOTE**: It is okay if 1 or 2 of the trials finish with "ERROR". This could happen from OOM issues but should not happen if you are running on ACEHUB. As long as 8 or 9 trials finish (status: TERMINATED) that is okay.

## Student Deliverables:
`lab5transfer.py`: 3 sets of confusion matrices with distinct file names and titles (output by `visualize_model_dataset`)
1. lab4 model as baseline CMs
2. transfer learning trained with frozen bottom layers CMs
3. transfer learning fine tuned with unfrozen layers CMs

`lab5ray.py`: 2 trial results tables, 2 sets of confusion matrices (1 set for each 'version' of search)
1. ASHA scheduler trial results table
2. ASHA scheduler best result CMs
3. Optuna trial results table
4. Optuna best result CMs

`heatmap.ipynb`: no deliverable