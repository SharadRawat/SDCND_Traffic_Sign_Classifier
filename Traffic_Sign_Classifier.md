
### This is my first attempt at writing markdown. Please leave some feedback about markdown editing or was of access as well. I will also be attaching the .html file. 


# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import os

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)


# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

#TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = []
for i in range(len(y_train)):
    if y_train[i] not in n_classes:
        n_classes.append(y_train[i])
n_classes = len(n_classes)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Personal Note: The summary of the dataset is mentioned above.

### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

### Personal Note:
 The dataset was explored and the images were plot randomly, one by one.
 A histogram was also plotted for train, valid and test data set. 
 It was observed that the data set was highly non-uniform. Some classes had a lot more examples than the other classes. This would have caused issues in the test accuracy of the classes that had a low frequency in the test data.
 In order to rectify this issue, few measures were taken:
 
 1) Augmentation using translation was used. (M = [1,0,0.5],[0,1,2])
 
 2) Augmentation using rotation was used. (Rotated 90 degrees about its centre.)
 
 3) After concatinating the augmented data to the train data. The first 2000 samples of each class were stored. This gave a more uniform-like histogram as can be seen in the cells below.


```python
img = X_train[2100] 
plt.imshow(img)
plt.show()
print(y_train[2100])
```


![png](output_11_0.png)


    23



```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

# Visualizations will be shown in the notebook.
img = X_train[0] 
plt.imshow(img)
plt.show()

unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Train Dataset Distribution")
plt.show()

unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_test, counts_test)
plt.grid()
plt.title("Test Dataset Distribution")
plt.show()

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(unique_valid, counts_valid)
plt.grid()
plt.title("Valid Dataset Distribution")
plt.show()

# Translation
more_X_train = []
more_y_train = []
for i in range(n_classes):
    count = 0
    for j in range(len(y_train)):
        if i == y_train[j] or count<=500:
            count = count + 1
            M = np.float32([[1,0,0.5],[0,1,2]])
            dst = cv2.warpAffine(X_train[j],M,(32,32))
            more_X_train.append(dst)
            more_y_train.append(i)
            
X_train = np.concatenate((X_train, more_X_train), axis=0)
y_train = np.concatenate((y_train, more_y_train), axis=0)

## Rotating
more_X_train = []
more_y_train = []
for i in range(n_classes):
    count = 0
    for j in range(len(y_train)):
        if i == y_train[j] and count<=500:
            count = count + 1
            M = cv2.getRotationMatrix2D((32/2,32/2),90,1)
            dst = cv2.warpAffine(X_train[j],M,(32,32))
            more_X_train.append(dst)
            more_y_train.append(i)
    
Sample_X = []
Sample_X.append(X_train[10000])
Sample_X.append(more_X_train[0])

X_train = np.concatenate((X_train, more_X_train), axis=0)
y_train = np.concatenate((y_train, more_y_train), axis=0)

Sample_X.append(more_X_train[0])

x_t=[]
y_t =[]
for i in range(n_classes):
    count = 0
    for j in range(len(y_train)):
        if i== y_train[j] and count <=2000:
            count+=1
            x_t.append(X_train[j])
            y_t.append(i)
    
    
#print(X_train[0])  
X_train = list(x_t)
y_train = list(y_t)



unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.title("Train Dataset Distribution after addition")
plt.show()

unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_test, counts_test)
plt.title("Test Dataset Distribution after addition")
plt.show()

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(unique_valid, counts_valid)
plt.title("Valid Dataset Distribution after addition")
plt.show()


%matplotlib inline
```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)



![png](output_12_3.png)



![png](output_12_4.png)



![png](output_12_5.png)



![png](output_12_6.png)



```python
print('After translation and rotation')
plt.imshow(Sample_X[1])



```

    After translation and rotation





    <matplotlib.image.AxesImage at 0x7fabd26afe80>




![png](output_13_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

### Personal Note:
Preprocessing consisted of:

Converting to grayscale - It helps to reduce training time.

Normalizing the data to the range (-1,1) - This was done using the line of code X_train_normalized = (X_train - 128)/128. This was done to make the imgae data centric for easy trining. SInce our countor plot of image is origin (or near origin) centric, gradient descent works efficiently. 


```python
import numpy as np
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.



# Convert to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

x_train =[]
X_valid = rgb2gray(X_valid)
X_test = rgb2gray(X_test)
for i in range(len(X_train)):
    x_train.append((rgb2gray(X_train[i]))/128-1)
    
X_train = x_train

X_valid = (X_valid -128)/128
X_test = (X_test -128)/128

plt.imshow(X_train[0],cmap='gray')


X_train = np.expand_dims(X_train, axis=3)
X_valid = np.expand_dims(X_valid, axis=3)
X_test = np.expand_dims(X_test, axis=3)

print(X_train[0].shape)


```

    (32, 32, 1)



![png](output_18_1.png)


### Personal Note:
An initial model architecture was used had the same layers as LeNet but with different filter sizes (3 instead of 5) with multiple iteration with strides. All the iteration gave really poor results. (20-50 ish validation accuracy. After 20-30 iterations, I moved to the Lenet architecture which gave me a booming 100% accuracy for training data (### This did not have augmented data) and 94% Validation accuracy.

### Important
Now, after I added the augmented data to the training set, the training accuracy reduced (70ish %) but, the validation accuracy imoproved and infact, it reached 90% very quickly. My test accuracy also improved. This is strange and I am still looking for proper answers to this (your input would be appriciated).

Another thig I noticed while analysing the data, 32x32 is a little small to capture the intricacies of the image. ex: train[2100] is a slipper road sign. I am not sure if this information was captured by the network (need your view on this too.)

### An attempt to this peculiarity
I thought since my model is rather underfit, thereore I added a 1x1 convolution layer with relu. This increased the number of parameters but the accuracy barely improved (around 71% ). This is still unsolved but since I met the criteria of 93% Valid_accuracy, I am submitting this architecure. No regularization was used as not required.

### Current Architecture
1) Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.

2) Pooling. Input = 28x28x6. Output = 14x14x6.

3) Layer 2: Convolutional. Output = 10x10x16

4) Pooling. Input = 10x10x16. Output = 5x5x16.

5) Layer 3: 1X1 convolution input = 5x5x16 output = 5x5x16

6) Flatten. Input = 5x5x16. Output = 400

7) Layer 4: Fully Connected. Input = 400. Output = 120.

8) Layer 5: Fully Connected. Input = 120. Output = 84.

9) Layer 6: Fully Connected. Input = 84. Output = 43.

### Model Architecture


```python
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

epochs = 41
learn_rate = 0.002
batch_size = 128

def traffic_net(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
     
    
    # 1X1 convolution input = 5x5x16 output = 5x5x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 16, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv3  = tf.nn.conv2d(conv2, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv3 = tf.nn.relu(conv3)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv3)
    
    
    
    # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 5: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 6: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

### Personal Note: 
Following optimizer and hyperparameters were used after numerous interations: 

Optimizer : Adam Optimizer becuase we dont have to worry about decaying learning rate. It is automatically reduced iteration by iteration.

epochs = 41

learn_rate = 0.002

batch_size = 128

mu =0

sigma = 0.1


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32,1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = traffic_net(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
training = tf.train.AdamOptimizer(learn_rate).minimize(loss)

prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_opt = tf.reduce_mean(tf.cast(prediction,tf.float32))

saver = tf.train.Saver()

def evaluate(x_data,y_data):
    num = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num,batch_size):
        x_batch,y_batch = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy= sess.run(accuracy_opt, feed_dict = {x:x_batch,y:y_batch})
        total_accuracy += (accuracy*len(x_batch))
    return total_accuracy/num

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num = len(X_train)
    print("Training...")
    print()
    validation_accuracy_figure = []
    training_accuracy_figure = []
    for i in range(epochs):
        X_train, y_train = shuffle(X_train,y_train)
        for offset in range(0,num,batch_size):
            X_data,y_data = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
            sess.run(training, feed_dict = {x:X_data,y:y_data})
        
        training_accuracy = evaluate(X_train,y_train)
        validation_accuracy = evaluate(X_valid,y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        validation_accuracy_figure.append(validation_accuracy)
        training_accuracy_figure.append(training_accuracy)
        
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

    test_accuracy = evaluate(X_test,y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Training...
    
    EPOCH 1 ...
    Training Accuracy = 0.607
    Validation Accuracy = 0.801
    
    EPOCH 2 ...
    Training Accuracy = 0.665
    Validation Accuracy = 0.854
    
    EPOCH 3 ...
    Training Accuracy = 0.685
    Validation Accuracy = 0.882
    
    EPOCH 4 ...
    Training Accuracy = 0.690
    Validation Accuracy = 0.884
    
    EPOCH 5 ...
    Training Accuracy = 0.696
    Validation Accuracy = 0.890
    
    EPOCH 6 ...
    Training Accuracy = 0.698
    Validation Accuracy = 0.888
    
    EPOCH 7 ...
    Training Accuracy = 0.698
    Validation Accuracy = 0.910
    
    EPOCH 8 ...
    Training Accuracy = 0.694
    Validation Accuracy = 0.893
    
    EPOCH 9 ...
    Training Accuracy = 0.701
    Validation Accuracy = 0.898
    
    EPOCH 10 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.915
    
    EPOCH 11 ...
    Training Accuracy = 0.702
    Validation Accuracy = 0.897
    
    EPOCH 12 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.930
    
    EPOCH 13 ...
    Training Accuracy = 0.700
    Validation Accuracy = 0.910
    
    EPOCH 14 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.920
    
    EPOCH 15 ...
    Training Accuracy = 0.704
    Validation Accuracy = 0.911
    
    EPOCH 16 ...
    Training Accuracy = 0.704
    Validation Accuracy = 0.916
    
    EPOCH 17 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.902
    
    EPOCH 18 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.907
    
    EPOCH 19 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.904
    
    EPOCH 20 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.920
    
    EPOCH 21 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.910
    
    EPOCH 22 ...
    Training Accuracy = 0.704
    Validation Accuracy = 0.909
    
    EPOCH 23 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.912
    
    EPOCH 24 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.921
    
    EPOCH 25 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.912
    
    EPOCH 26 ...
    Training Accuracy = 0.707
    Validation Accuracy = 0.914
    
    EPOCH 27 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.921
    
    EPOCH 28 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.905
    
    EPOCH 29 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.910
    
    EPOCH 30 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.918
    
    EPOCH 31 ...
    Training Accuracy = 0.704
    Validation Accuracy = 0.915
    
    EPOCH 32 ...
    Training Accuracy = 0.706
    Validation Accuracy = 0.938
    
    EPOCH 33 ...
    Training Accuracy = 0.707
    Validation Accuracy = 0.929
    
    EPOCH 34 ...
    Training Accuracy = 0.703
    Validation Accuracy = 0.924
    
    EPOCH 35 ...
    Training Accuracy = 0.698
    Validation Accuracy = 0.897
    
    EPOCH 36 ...
    Training Accuracy = 0.707
    Validation Accuracy = 0.930
    
    EPOCH 37 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.918
    
    EPOCH 38 ...
    Training Accuracy = 0.707
    Validation Accuracy = 0.926
    
    EPOCH 39 ...
    Training Accuracy = 0.708
    Validation Accuracy = 0.929
    
    EPOCH 40 ...
    Training Accuracy = 0.705
    Validation Accuracy = 0.911
    
    EPOCH 41 ...
    Training Accuracy = 0.707
    Validation Accuracy = 0.927
    
    Model saved
    Test Accuracy = 0.909


### Personal Note
To test the model's accuracy, test data was used and was run through the evaluate() function. Since no dropout was used, there was no need to change anything in the model. A good test accuracy of 91% was attained.


```python
plt.plot(training_accuracy_figure)
plt.title("training_accuracy")
plt.show()

plt.plot(validation_accuracy_figure)
plt.title("validation_accuracy")
plt.show()
```


![png](output_27_0.png)



![png](output_27_1.png)


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Personal Note:
New images- hand picked from internet.

All the images should be classified. But there can be issues, becuase some images have text(watermark on them). This could be a reason for poor classification.


### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob
import cv2

my_images = sorted(glob.glob('./test_images/*.jpg'))
my_labels = np.array([1, 22, 34, 33, 14])
my_signs = []
my_image = 0
fig = plt.figure()
for my_image in my_images:
    img = cv2.cvtColor(cv2.imread(my_image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32)) 
    img = img/128-1
    my_signs.append(img)
    plt.figure()
    fig.add_subplot(i,2,1)
    plt.imshow(cv2.imread(my_image))
    
    

my_signs = np.expand_dims(my_signs, axis=3)




```


![png](output_31_0.png)



![png](output_31_1.png)



![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_accuracy = evaluate(my_signs, my_labels)
    print("Data Set Accuracy = {:.3f}".format(my_accuracy))
```

    Data Set Accuracy = 0.400


### Personal Note: 
Accuracy is poorer than the test set accuracy. The model was unable to classify real world images. 

However, on checking the softmax probablities (top_5), it was observed that for 4 out of 5 images, the correct classification was in the top 5 predictions. This would mean that, more data would be required for the training of those classes.

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
single_image = []
single_label = []

for i in range(5):
    single_image.append(my_signs[i])
    single_label.append(my_labels[i])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#         saver = tf.train.import_meta_graph('./lenet.meta')
        saver.restore(sess, "./lenet")
        my_accuracy = evaluate( single_image, single_label)
        print('Image {}')
        print("Image Accuracy = {:.3f}".format(my_accuracy))
        
```

    Image {}
    Image Accuracy = 0.000
    Image {}
    Image Accuracy = 0.500
    Image {}
    Image Accuracy = 0.333
    Image {}
    Image Accuracy = 0.250
    Image {}
    Image Accuracy = 0.400


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#         saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    softmax_logits = sess.run(tf.nn.softmax(logits),feed_dict ={x: my_signs})
    top_5 = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5),feed_dict ={x: my_signs})
    print(top_5)
# ACTUAL LABELS [1, 22, 34, 33, 14]

    fig, axs = plt.subplots(len(my_images),4, figsize=(12, 14))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()
    image = 0

    for i, image in enumerate(my_images):
        axs[4*i].axis('off')
        axs[4*i].imshow(plt.imread(image))
        axs[4*i].set_title('input')
        guess1 = top_5[1][i][0]
        index1 = np.argwhere(y_valid == guess1)[0]
        axs[4*i+1].axis('off')
        axs[4*i+1].imshow(X_valid[index1].squeeze(), cmap='gray')
        axs[4*i+1].set_title('top guess: {} '.format(guess1))
        guess2 = top_5[1][i][1]
        index2 = np.argwhere(y_valid == guess2)[0]
        axs[4*i+2].axis('off')
        axs[4*i+2].imshow(X_valid[index2].squeeze(), cmap='gray')
        axs[4*i+2].set_title('2nd guess: {}'.format(guess2))
        guess3 = top_5[1][i][2]
        index3 = np.argwhere(y_valid == guess3)[0]
        axs[4*i+3].axis('off')
        axs[4*i+3].imshow(X_valid[index3].squeeze(), cmap='gray')
        axs[4*i+3].set_title('3rd guess: {}'.format(guess3))
```

    TopKV2(values=array([[  9.71445441e-01,   2.83010881e-02,   2.03102842e-04,
              4.26264596e-05,   7.50476738e-06],
           [  9.99999046e-01,   9.01500982e-07,   8.35545744e-10,
              5.93916305e-10,   3.57526834e-13],
           [  9.75794435e-01,   1.23829730e-02,   5.69843547e-03,
              4.46118554e-03,   8.42648966e-04],
           [  3.49588364e-01,   2.20666990e-01,   8.74108672e-02,
              5.54014407e-02,   5.36559820e-02],
           [  1.00000000e+00,   2.70814837e-08,   8.18879193e-18,
              1.40374408e-20,   1.92460465e-22]], dtype=float32), indices=array([[ 5,  1,  0,  2,  3],
           [22, 34, 35, 29, 36],
           [10, 34, 25, 42, 37],
           [ 8,  7, 16,  5, 11],
           [14,  4, 38, 39,  0]], dtype=int32))



![png](output_39_1.png)



```python
fig, axs = plt.subplots(5,2, figsize=(9, 19))
axs = axs.ravel()

for i in range(len(softmax_logits)*2):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.imread(my_images[i//2]))
    else:
        axs[i].bar(np.arange(n_classes), softmax_logits[(i-1)//2]) 
        axs[i].set_ylabel('Softmax probability')
# ACTUAL LABELS [3, 22, 34,9, 33, 14]
```


![png](output_40_0.png)


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#         saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    image_input = X_train[0]
    tf_activation = tf.Variable(conv2)
    outputFeatureMap(image_input,tf_activation)

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

    def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
        activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
        featuremaps = activation.shape[3]
        plt.figure(plt_num, figsize=(15,15))
        for featuremap in range(featuremaps):
            plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
            plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
            if activation_min != -1 & activation_max != -1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
            elif activation_max != -1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
            elif activation_min !=-1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
            else:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-189-3dc68972c6a5> in <module>()
          6     saver.restore(sess, "./lenet")
          7     image_input = X_train[0]
    ----> 8     tf_activation = tf.Variable(conv2)
          9     outputFeatureMap(image_input,tf_activation)
         10 


    NameError: name 'conv2' is not defined


### Personal Note: 
I tried the step 4. But was clueless. I would appreciate if imputs can be given here as well. 
(Semester Final exam week going on, that's why did not have a lot of time for Step4).





```python

```
