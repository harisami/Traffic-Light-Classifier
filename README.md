{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Traffic Light Classifier\n",
    "---\n",
    "\n",
    "In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.\n",
    "\n",
    "In this notebook, you'll pre-process these images, extract features that will help us distinguish the different types of images, and use those features to classify the traffic light images into three classes: red, yellow, or green. The tasks will be broken down into a few sections:\n",
    "\n",
    "1. **Loading and visualizing the data**. \n",
    "      The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!\n",
    "\n",
    "2. **Pre-processing**. \n",
    "    The input images and output labels need to be standardized. This way, you can analyze all the input images using the same classification pipeline, and you know what output to expect when you eventually classify a *new* image.\n",
    "    \n",
    "3. **Feature extraction**. \n",
    "    Next, you'll extract some features from each image that will help distinguish and eventually classify these images.\n",
    "   \n",
    "4. **Classification and visualizing error**. \n",
    "    Finally, you'll write one function that uses your features to classify *any* traffic light image. This function will take in an image and output a label. You'll also be given code to determine the accuracy of your classification model.    \n",
    "    \n",
    "5. **Evaluate your model**.\n",
    "    To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!\n",
    "    \n",
    "Here are some sample images from the dataset (from left to right: red, green, and yellow traffic lights):\n",
    "<img src=\"images/all_lights.png\" width=\"50%\" height=\"50%\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### *Here's what you need to know to complete the project:*\n",
    "\n",
    "Some template code has already been provided for you, but you'll need to implement additional code steps to successfully complete this project. Any code that is required to pass this project is marked with **'(IMPLEMENTATION)'** in the header. There are also a couple of questions about your thoughts as you work through this project, which are marked with **'(QUESTION)'** in the header. Make sure to answer all questions and to check your work against the [project rubric](https://review.udacity.com/#!/rubrics/1213/view) to make sure you complete the necessary classification steps!\n",
    "\n",
    "Your project submission will be evaluated based on the code implementations you provide, and on two main classification criteria.\n",
    "Your complete traffic light classifier should have:\n",
    "1. **Greater than 90% accuracy**\n",
    "2. ***Never* classify red lights as green**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Loading and Visualizing the Traffic Light Dataset\n",
    "\n",
    "This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:\n",
    "* 904 red traffic light images\n",
    "* 536 green traffic light images\n",
    "* 44 yellow traffic light images\n",
    "\n",
    "*Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import resources\n",
    "\n",
    "Before you get started on the project code, import the libraries and resources that you'll need."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2 # computer vision library\n",
    "import helpers # helper functions\n",
    "\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg # for loading in images\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training and Testing Data\n",
    "\n",
    "All 1484 of the traffic light images are separated into training and testing datasets. \n",
    "\n",
    "* 80% of these images are training images, for you to use as you create a classifier.\n",
    "* 20% are test images, which will be used to test the accuracy of your classifier.\n",
    "* All images are pictures of 3-light traffic lights with one light illuminated.\n",
    "\n",
    "## Define the image directories\n",
    "\n",
    "First, we set some variables to keep track of some where our images are stored:\n",
    "\n",
    "    IMAGE_DIR_TRAINING: the directory where our training image data is stored\n",
    "    IMAGE_DIR_TEST: the directory where our test image data is stored"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Image data directories\n",
    "IMAGE_DIR_TRAINING = \"traffic_light_images/training/\"\n",
    "IMAGE_DIR_TEST = \"traffic_light_images/test/\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load the datasets\n",
    "\n",
    "These first few lines of code will load the training traffic light images and store all of them in a variable, `IMAGE_LIST`. This list contains the images and their associated label (\"red\", \"yellow\", \"green\"). \n",
    "\n",
    "You are encouraged to take a look at the `load_dataset` function in the helpers.py file. This will give you a good idea about how lots of image files can be read in from a directory using the [glob library](https://pymotw.com/2/glob/). The `load_dataset` function takes in the name of an image directory and returns a list of images and their associated labels. \n",
    "\n",
    "For example, the first image-label pair in `IMAGE_LIST` can be accessed by index: \n",
    "``` IMAGE_LIST[0][:]```.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using the load_dataset function in helpers.py\n",
    "# Load training data\n",
    "IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize the Data\n",
    "\n",
    "The first steps in analyzing any dataset are to 1. load the data and 2. look at the data. Seeing what it looks like will give you an idea of what to look for in the images, what kind of noise or inconsistencies you have to deal with, and so on. This will help you understand the image dataset, and **understanding a dataset is part of making predictions about the data**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Visualize the input images\n",
    "\n",
    "Visualize and explore the image data! Write code to display an image in `IMAGE_LIST`:\n",
    "* Display the image\n",
    "* Print out the shape of the image \n",
    "* Print out its corresponding label\n",
    "\n",
    "See if you can display at least one of each type of traffic light image – red, green, and yellow — and look at their similarities and differences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image Dimensions:  (91, 38, 3)\n",
      "Image Label:  red\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f29b1e72400>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f29b3eeeba8>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)\n",
    "## TODO: Print out 1. The shape of the image and 2. The image's label\n",
    "\n",
    "# The first image in IMAGE_LIST is displayed below (without information about shape or label)\n",
    "num = 0\n",
    "selected_image = IMAGE_LIST[num][0]\n",
    "selected_label = IMAGE_LIST[num][1]\n",
    "\n",
    "# i = 0\n",
    "# for image in IMAGE_LIST:\n",
    "#     i = i + 1\n",
    "#     if image[1] == \"yellow\":\n",
    "#         print(\"Image Dimensions: \", image[0].shape)\n",
    "#         print(\"Image Label: \", image[1])\n",
    "#         print(\"File Number: \", i)\n",
    "#         plt.imshow(image[0])\n",
    "\n",
    "\n",
    "print(\"Image Dimensions: \", selected_image.shape)\n",
    "print(\"Image Label: \", selected_label)\n",
    "plt.imshow(selected_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Pre-process the Data\n",
    "\n",
    "After loading in each image, you have to standardize the input and output!\n",
    "\n",
    "### Input\n",
    "\n",
    "This means that every input image should be in the same format, of the same size, and so on. We'll be creating features by performing the same analysis on every picture, and for a classification task like this, it's important that **similar images create similar features**! \n",
    "\n",
    "### Output\n",
    "\n",
    "We also need the output to be a label that is easy to read and easy to compare with other labels. It is good practice to convert categorical data like \"red\" and \"green\" to numerical data.\n",
    "\n",
    "A very common classification output is a 1D list that is the length of the number of classes - three in the case of red, yellow, and green lights - with the values 0 or 1 indicating which class a certain image is. For example, since we have three classes (red, yellow, and green), we can make a list with the order: [red value, yellow value, green value]. In general, order does not matter, we choose the order [red value, yellow value, green value] in this case to reflect the position of each light in descending vertical order.\n",
    "\n",
    "A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.\n",
    "\n",
    "*(Note: one-hot encoding will be especially important when you work with [machine learning algorithms](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)).*\n",
    "\n",
    "<img src=\"images/processing_steps.png\" width=\"80%\" height=\"80%\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='task2'></a>\n",
    "### (IMPLEMENTATION): Standardize the input images\n",
    "\n",
    "* Resize each image to the desired input size: 32x32px.\n",
    "* (Optional) You may choose to crop, shift, or rotate the images in this step as well.\n",
    "\n",
    "It's very common to have square input sizes that can be rotated (and remain the same size), and analyzed in smaller, square patches. It's also important to make all your images the same size so that they can be sent through the same pipeline of classification steps!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function should take in an RGB image and return a new, standardized version\n",
    "def standardize_input(image):\n",
    "    \n",
    "    ## TODO: Resize image and pre-process so that all \"standard\" images are the same size  \n",
    "    standard_im = np.copy(image)\n",
    "    \n",
    "    standard_im = cv2.resize(image, (32, 32))\n",
    "    \n",
    "    return standard_im"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Standardize the output\n",
    "\n",
    "With each loaded image, we also specify the expected output. For this, we use **one-hot encoding**.\n",
    "\n",
    "* One-hot encode the labels. To do this, create an array of zeros representing each class of traffic light (red, yellow, green), and set the index of the expected class number to 1. \n",
    "\n",
    "Since we have three classes (red, yellow, and green), we have imposed an order of: [red value, yellow value, green value]. To one-hot encode, say, a yellow light, we would first initialize an array to [0, 0, 0] and change the middle value (the yellow value) to 1: [0, 1, 0].\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='task3'></a>\n",
    "### (IMPLEMENTATION): Implement one-hot encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: One hot encode an image label\n",
    "## Given a label - \"red\", \"green\", or \"yellow\" - return a one-hot encoded label\n",
    "\n",
    "# Examples: \n",
    "# one_hot_encode(\"red\") should return: [1, 0, 0]\n",
    "# one_hot_encode(\"yellow\") should return: [0, 1, 0]\n",
    "# one_hot_encode(\"green\") should return: [0, 0, 1]\n",
    "\n",
    "def one_hot_encode(label):\n",
    "    \n",
    "    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights\n",
    "    one_hot_encoded = [0, 0, 0]\n",
    "    \n",
    "    color_map = ['red', 'yellow', 'green']\n",
    "    \n",
    "    if label not in color_map:\n",
    "        raise TypeError(\"Please input a valid label: (red, yellow, green)\")\n",
    "    \n",
    "    elif label == \"red\":\n",
    "        one_hot_encoded[0] = 1\n",
    "    elif label == \"yellow\":\n",
    "        one_hot_encoded[1] = 1\n",
    "    else: # green\n",
    "        one_hot_encoded[2] = 1\n",
    "    \n",
    "    return one_hot_encoded"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Testing as you Code\n",
    "\n",
    "After programming a function like this, it's a good idea to test it, and see if it produces the expected output. **In general, it's good practice to test code in small, functional pieces, after you write it**. This way, you can make sure that your code is correct as you continue to build a classifier, and you can identify any errors early on so that they don't compound.\n",
    "\n",
    "All test code can be found in the file `test_functions.py`. You are encouraged to look through that code and add your own testing code if you find it useful!\n",
    "\n",
    "One test function you'll find is: `test_one_hot(self, one_hot_function)` which takes in one argument, a one_hot_encode function, and tests its functionality. If your one_hot_label code does not work as expected, this test will print ot an error message that will tell you a bit about why your code failed. Once your code works, this should print out TEST PASSED."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "**<span style=\"color: green;\">TEST PASSED</span>**"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Importing the tests\n",
    "import test_functions\n",
    "tests = test_functions.Tests()\n",
    "\n",
    "# Test for one_hot_encode function\n",
    "tests.test_one_hot(one_hot_encode)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Construct a `STANDARDIZED_LIST` of input images and output labels.\n",
    "\n",
    "This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.\n",
    "\n",
    "This uses the functions you defined above to standardize the input and output, so those functions must be complete for this standardization to work!\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "def standardize(image_list):\n",
    "    \n",
    "    # Empty image data array\n",
    "    standard_list = []\n",
    "\n",
    "    # Iterate through all the image-label pairs\n",
    "    for item in image_list:\n",
    "        image = item[0]\n",
    "        label = item[1]\n",
    "\n",
    "        # Standardize the image\n",
    "        standardized_im = standardize_input(image)\n",
    "\n",
    "        # One-hot encode the label\n",
    "        one_hot_label = one_hot_encode(label)    \n",
    "\n",
    "        # Append the image, and it's one hot encoded label to the full, processed list of image data \n",
    "        standard_list.append((standardized_im, one_hot_label))\n",
    "        \n",
    "    return standard_list\n",
    "\n",
    "# Standardize all training images\n",
    "STANDARDIZED_LIST = standardize(IMAGE_LIST)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize the standardized data\n",
    "\n",
    "Display a standardized image from STANDARDIZED_LIST and compare it with a non-standardized image from IMAGE_LIST. Note that their sizes and appearance are different!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Image Size:  (91, 38, 3)\n",
      "Original Image Label:  red\n",
      "Standardized Image Size:  (32, 32, 3)\n",
      "Standardized Image Label:  [1, 0, 0]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f29b1d84240>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f29b1d94320>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## TODO: Display a standardized image and its label\n",
    "num = 0\n",
    "standardized_image = STANDARDIZED_LIST[num][0]\n",
    "standardized_label = STANDARDIZED_LIST[num][1]\n",
    "\n",
    "\n",
    "f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))\n",
    "print(\"Original Image Size: \", selected_image.shape)\n",
    "print(\"Original Image Label: \", selected_label)\n",
    "print(\"Standardized Image Size: \", standardized_image.shape)\n",
    "print(\"Standardized Image Label: \", standardized_label)\n",
    "ax1.set_title(\"Standardized Image\")\n",
    "ax1.imshow(standardized_image)\n",
    "ax2.set_title(\"Original Image\")\n",
    "ax2.imshow(selected_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Feature Extraction\n",
    "\n",
    "You'll be using what you now about color spaces, shape analysis, and feature construction to create features that help distinguish and classify the three types of traffic light images.\n",
    "\n",
    "You'll be tasked with creating **one feature** at a minimum (with the option to create more). The required feature is **a brightness feature using HSV color space**:\n",
    "\n",
    "1. A brightness feature.\n",
    "    - Using HSV color space, create a feature that helps you identify the 3 different classes of traffic light.\n",
    "    - You'll be asked some questions about what methods you tried to locate this traffic light, so, as you progress through this notebook, always be thinking about your approach: what works and what doesn't?\n",
    "\n",
    "2. (Optional): Create more features! \n",
    "\n",
    "Any more features that you create are up to you and should improve the accuracy of your traffic light classification algorithm! One thing to note is that, to pass this project you must **never classify a red light as a green light** because this creates a serious safety risk for a self-driving car. To avoid this misclassification, you might consider adding another feature that specifically distinguishes between red and green lights.\n",
    "\n",
    "These features will be combined near the end of his notebook to form a complete classification algorithm."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating a brightness feature \n",
    "\n",
    "There are a number of ways to create a brightness feature that will help you characterize images of traffic lights, and it will be up to you to decide on the best procedure to complete this step. You should visualize and test your code as you go.\n",
    "\n",
    "Pictured below is a sample pipeline for creating a brightness feature (from left to right: standardized image, HSV color-masked image, cropped image, brightness feature):\n",
    "\n",
    "<img src=\"images/feature_ext_steps.png\" width=\"70%\" height=\"70%\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## RGB to HSV conversion\n",
    "\n",
    "Below, a test image is converted from RGB to HSV colorspace and each component is displayed in an image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label [red, yellow, green]: [0, 0, 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f29b1c339b0>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABIEAAAEiCAYAAABuhsImAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAIABJREFUeJzt3XucZXdZ5/vvs3fduruq0ulLha6kY8CmI8KJQSJ6vDCMdx0c8HUcXug5iB5HZpzh5HBUXsNhDgrGGdHXIGM83kCQoEjAARU9qCCjIDMOEghBQkwTIKZDNel7qqqr67p/54+9O1Q6tZ5n7/Xbu/aqXp/369WvVOqp31rP+q21n1r7V3vvx1JKAgAAAAAAwJWtMewEAAAAAAAAMHgsAgEAAAAAANQAi0AAAAAAAAA1wCIQAAAAAABADbAIBAAAAAAAUAMsAgEAAAAAANQAi0BXODN7rpk93Mft3WBmycxGOv//Z2b2kn5tv7PN15jZ7xXEvsXM7u/n/gDsHF59GCYz+2sz+5fDzgPA9ujcCx0Zdh6b9fueD0D1VfVxX9X7NbSxCLQNzOybzey/m9mjZnbWzP6bmX1dJ/YjZvaRYedYVkrpe1JKd2zj/v4mpXTjdu0PQD4ze9DMvv2y7+3o2gdgZ/LuyQBgO5jZX5jZz23x/eeb2Zcu/bEdGBQWgQbMzKYl/amkX5W0T9K1kl4raWWYeXWDAgQAAK4UO/meDMAV5a2SXmxmdtn3Xyzp7Sml9e1PCXXCItDgHZWklNI7UkobKaWLKaX3p5Q+ZWZPk/Sbkv5nM1s0s/OSZGb/zMzuNrN5MztuZq+5tLFNb8d6iZk9ZGanzezfb4rvMrO3mtk5M/uMpMf9dcvMXmlmnzOzBTP7jJl9/6bYj3T+IvYGMzsr6TVm1jSz/9TZz+cl/bPLtvfYWyDM7J7OcVz6l8zsuZ3YN3T+8na+83PP3bSNJ5vZhzo5fUDSgaLJvPwlj51XGLzCzD5lZhfM7M1mdk3nbWoLZvaXZnb1pp//g84K+6Nm9mEze/qm2H4z+5POvH/MzH5+8ysVzOyrzOwDnb8c3m9mLyzKE0AeM3v6psfbI2b2qk3hMTN7W+cxfq+Z3bJpXFTjPtKpaefM7Atm9j2b4n9tZrd16uCCmb3fzA5sihfWMQA7QuE92VY/3LkHetWmmvJxMzu86Ue+3cw+26knv3bpCZ2ZfaWZ/VczO9O5f3q7me3dtN0HzeynO/cuj5rZO81sohN7rpk9bGY/ZWYnzeyEmf3oprHjnRr2UKc2/qaZ7RrIbAEYlD9SeyH6Wy59o/N85XmS3rbVADPbZ2a/Y2ZznZrzR5fFi2pGzvPK15jZu5x7rlkze7eZnercU93ah7nBNmARaPCOSdowszvM7Hs2L0iklO6T9K8l/W1KaTKldOkG4YKkH5a0V+1Fl58wsxdctt1vlnSjpG+T9DPWXlCSpJ+V9JWdf98l6fLP6/mc2gXnKrX/+vV7ZnZoU/zrJX1e0oyk/yDpx9UuSM+UdIukHyg60JTS13SOY1LST0q6X9InzOxaSf+fpJ9Xu+D9tKR3m9nBztDfl/RxtRd/btsi58j/Iuk71L65+z5JfybpVZ3tNSRtLkh/JumpneP7hKS3b4r9mtpz/6RODo/lYWZ7JH2gk+uMpB+U9OubF5EA9IeZTUn6S0l/LmlW0hFJH9z0I/9c0p1q18j3Svp/N8W6qXH3q10ffknSmy89cev4IUk/qvbjfEzteqUu6hiA6iu8Jyvwk2r/vv9eSdOS/ndJS5viz1P7j21fI+mFat93SZJJ+gW169fTJB2W9JrLtv1CSd8t6cmSbpL0I5tiT1K7hl0r6cck/dqmXH9R7fudm9WujddK+pngOABUSErpoqR3qf1875IXSvqHlNI9BcN+V9JuSU9X+x7lDZtiXs3IeV4pFdxzmVlD0p9Iuqez32+T9HIz+y6h8lgEGrCU0rzaD6wk6U2STpnZe83sGmfMX6eU/j6l1Or8deodkv7JZT/22s5fsO5R+8H3NZ3vv1DSf0gpnU0pHZd0+2Xb/oOU0lxn2++U9FlJz970I3MppV9NKa13CtQLJf3nlNLxlNJZtW9qXGb2zWo/UfrnneP/3yS9L6X0vs5+PyDpLknfa2bXq30D9eqU0kpK6cNqF5Re/GpK6ZGU0hcl/Y2kj6aU7k4prUj6Q7UXsC4d/1tSSgud2GskfY2ZXWVmTbUXk342pbSUUvqMpM2fdfQ8SQ+mlH6nMzefkPRuOYtiAB7njzqvoDlv7Vc9/rrzs8+T9KWU0utTSsudx+xHN8U/0qknG2rfFF2qf93UuH9MKb2pM/YOSYckba7Hv5NSOrbpBu3mzvcL61ip2QCw7Urck/1LSf9PSun+1HZPSunMpvjrUkrnU0oPSfordepFSumBlNIHOvc1pyT9sp54H3d7p1adVfu+5+ZNsTVJP5dSWkspvU/SoqQbOwvWPy7p/+rc5y1I+o+SXlR+VgAMyR2S/sWmV/L9sB7/3OMxnT9mfY+kf51SOtepDR/a9CNb1gwp+3mlVHzP9XWSDqaUfi6ltJpS+rzadZV6tAOwCLQNUkr3pZR+JKV0naRnqP2Xof9c9PNm9vVm9ledl9Y9qvarhS5/i9SXNn29JGmy8/WspOObYv942bZ/2Mw+uemJ2DMu2/bmseH2tsj9sNpPnF6SUjrW+fZXqF3kNj8B/Ga1n3zNSjqXUrrQ7T628Mimry9u8f+TndyaZva6zsu65yU92PmZA5IOShrR449189dfIenrLzuG/1XtlXcAsReklPZe+ifp3zg/e1jtV/QUubz+TdiXOxZGNe6xsSmlS3/Rn9wqrsfXVq+OAdgherwn67UWXbrfmDGzO83si537jd9T9/dxknTmss8EuRQ/qPYrAT6+qQ79eef7AHaQlNJHJJ2S9Hwze4raiyq/X/DjhyWdTSmdK4gX1Yzc55VbxS7dc32FpNnL7otepcf/YQ0VxSLQNksp/YPaHwb2jEvf2uLHfl/tl9sdTildpfbnBl3+wWFFTqhdKC65/tIXZvYVaq/QvkzS/s4TsU9ftu3L8ync3uU6K9l/pPYrh/5sU+i4pN/d/AQwpbQnpfS6zvav7rzdKtxHph+S9HxJ3672SyZvuJS62kV4XdJ1m35+83Efl/Shy45hMqX0EwPKFaiz42q/pbUnXda4nJyK6hiAHWiLe7LLlapFar9qOkm6KaU0rfYrCftRh06r/cetp2+qQ1el9tvwAew8b1P7FUAvlvT+lNIjBT93XNI+2/TZYj3IeV7pOS7pC5fdF02llHiF9A7AItCAWfvDhH/KzK7r/P9htd9f/j86P/KIpOvMbGzTsCm1V3uXzezZai9edOtdkv5vM7u6s8//Y1Nsj9o3Jac6ufyoim98Nm/vVjO7rvPe0lc6P/sWtd/L+kuXff/3JH2fmX1X59U4E9b+4MPrUkr/qPZbKl5rZmOdt5J9X7cH26MptTuAnFH7L2n/8VKg8xLH96j9Ydi7zeyr9Pj36f6ppKNm9mIzG+38+7rL3jMLoD/+VNKTzOzl1v4Q1Ckz+/ouxpWpcd0qrGN92j6AAevinuxyvy3pNjN7qrXdZGb7u9jVlNpvxzjf+TyxV/Qj/5RSS+2F7jeY2YzU/rwyPoMD2LHepvYfp39cBW8Fk6SU0gm1P9f01zvP8UbN7Dld7iPneaXn7yTNm9m/s3ZjoqaZPcPMvi4ciaFjEWjwFtT+INKPmtkFtW80Pi3ppzrx/yrpXklfMrPTne/9G0k/Z2YLan/Y37t62N9r1X471RckvV/t925Kkjqfc/N6SX+r9uLT/yTpvwXbe5Okv1D7/aGfUHuhpMiLJH2/Pb5D2Lek9mcTPV/tlwieUnvl+BX68vX3Q2rP0Vm1P9h6y0/F74O3qT03X5T0GT3xpu9lar9C6Etqz9s71Gkb23nf/Xd2jnGu8zO/KGl8QLkCtdV5vH2H2gvCX1L7c33+aRfjytS4bnOK6hiA6ovuyS73y2rfg71f0rykN0vqphPXayV9raRH1f5Aee/eqVf/TtIDkv5H561mf6nOZ38A2FlSSg9K+u9q/xHrvcGPv1jtz/75B0knJb28y93kPK8s1PkD+vep/XlmX1D7lYq/rfZzKVScpbTVu5EAmNkvSnpSSqnXbmUAAAAAAFQOf8EEOjovE7+p85LvZ6vdYvEPh50XAAAAAAD9MDLsBIAKmVL7LWCzar/M8vWS/nioGQEAAAAA0Ce8HQwAAAAAAKAGeDsYAAAAAABADbAIBAAAAAAAUANZnwlkZt8t6VckNSX9dkrpdd7P79+/Px2+7nDOLh3e29rMHxqE/d1Gb6fL2fgAN+2m7R+TWfHaYSu13LHra2tufKNVPL7ZbLpjR5rFl7PZEM9DxqVZVffcc8/plNLBYedxSa+16MCBA+mGG27YjtQqo+U8thqNvL8HeG8rznnL8d133+3Gn/nMZxbGvOOV8o8Z1XD33XdXqhZJvdWjkZGRND4+vm25Xek2NjYKY1FNKLtdyb8/iWpNTq3KOSb019ra2o6uRXv37k2zs7Pbllu3cu7fo8fHQJ8bZMi5pxoZKX4utBY8Bzt//rwbX15eLozt3r3bHTs1NVUYi57f5YjOsTef0dic6yfn3jja72c+85mualHpRSAza0r6NUnfIelhSR8zs/emlD5TNObwdYf1wfe/v+wuXcM6idGNgbdgkl17nA1El5ZbYILRY6NjhbHli8UFQpIeOfmIG19YWCiMXXXVVe7Y/fv3F8ZGR0fdsTkGWWBynqjm/PKL8tp/8MA/lkpqAMrUohtuuEEf+9jHtivFSlhcXCyMTU5OZm17fX29MObdNESmp6fd+Ic+9KHC2MrKijuWJ95Xhunp6crUIqn3ejQ+Pq6nPe1p25niFc27h/BiOduV/Cc4Xix32znHhP46ceLEjq5Fs7Ozevvb3z6QXLznSuEfeJ1FjcjS0pIb9/Y9yIWJ6LmjF48Wcg4eLH7uPzc35479kz/5Ezd+3333Fca8P8pJ0nOe85zC2NVXX+2OzRGdx5xrM+c+MrpH9fYdPSZuuummrmpRzp9Cny3pgZTS51NKq5LulPT8jO0BQBnUIgBVQT0CUAXUIgCFchaBrpV0fNP/P9z5HgBsJ2oRgKqgHgGoAmoRgEI5i0BbvV/kCe97MbOXmtldZnbXmbNnMnYHAFvquRadOnVqG9ICUENhPdpci7y3UgJAhp5q0blz57YpLQBVkLMI9LCkzZ/yfJ2kJ7zZMKX0xpTSLSmlW/bvK/7MFgAoqeda5L1nGgAyhPVocy3K+bwLAHD0VIsG+bksAKonZxHoY5KeamZPNrMxSS+S9N7+pAUAXaMWAagK6hGAKqAWAShU+k9QKaV1M3uZpL9Qu/XgW1JK9/YtMwDoArUIQFVQjwBUAbUIgCfrdcgppfdJel9vg5xYRst0r6V11Gbba+Et+a22c1qPB7sNW8i3og04vDmJXp7utXxeWPTblV4MWsh7LaH37dtXemzULj36XIZovMe9/qKTnCFq8+5dgDnX1jD0WotSSm57xiuxfbjXBt5rHx+NHaT5+fnSY6NzePLkSTc+MzNTet870TDnY2JiojDm/b6pql7qUaPRoAX4DhC1effktIDvZvyVJne+clxpj8VSz9MGwGt3Hf2ujtqpe/dye/fu9RPL2G/UPnx1dbX0vr05ia7/hx9+uDB2773+GqA3VpJuuummwtg3fuM3lh4bzVX02IvarXvGxsZKj80RXT+enOPdLOftYAAAAAAAANghWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACogawW8b0y89tlt1L5NtyeqM1fxGshH7W18+LR2LW1NTe+4eQVHbMXj1qLe3nlHtPISHHLvAsXLrhjk4rnw+QfU841ErX5Gxkpfpi57ePVRZt357gaUYd4b9s7rEU88uS2gN+JbbwH2fJ8enq6MJbT9n6QcufDazF//fXXu2N34vWDwcht/z03N9enTHpT1bblXnyQOecY1lxKO7MNfL80Gg23NXm/2lJfbmlpKWu8d/9+6tQpd+zZs2cLY6dPn3bHnj9/3o17xxUdsxePnjd4eZ05c6b0WMm/V/zc5z7njm21ip/jR8d08eJFN+7ZtWuXG/fu17zHg5T33DHa9vr6eultd4tXAgEAAAAAANQAi0AAAAAAAAA1wCIQAAAAAABADbAIBAAAAAAAUAMsAgEAAAAAANQAi0AAAAAAAAA1wCIQAAAAAABADYxs585SklqpVRhvtYpjkdHRUWe/yR0b7dcb3mz662hnzpwpFZOkVpD3yEjx6Ws0m+5YMyveb8Z5iOY68uj8o4WxxcVFd6w3H81gPryxktRoFJ9nLxZte/fu3e7YXbt2ufHx8fHCWM6ZiI5ppzMzd+6qynsMTE5ODmy/6+vrbnxiYqIwtry83O90tsX09HRhbH5+3h0bxa9EMzMzhbGdeg1sh1arpYWFhWGnsa2mpqZKj43manZ2tvTYqpqbmyuMeccr5c11jqrOdVXzqoJWq6WVlZXC+Orqault7927tzC2sbHhjvVyisbv2bPHHfue97ynMPY3f/M37tjo99pVV11VGBsbG3PHes9Zcs5DNNeRu+++uzB27Ngxd6xXi6LnOt79mOTPZ3Sv7+X15Cc/2R17+PBhN37NNdcUxqL7ak+/nr9c2c/0AAAAAAAAIIlFIAAAAAAAgFpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBrY1hbxUnJbiEdtvD3rXtu7oG151A67OVLcTn1tdc0d67Vij1oEhu3WnbyjsV4beC/nSDg2Ohcjo6X37bU+jNoi5rS2X1vzr4EcUQt5r01glJd3DaA8r+3jyEheyfXaZA7yfEZ5e63rq8pray/Vs807rixeO+zc1uGD3LYn2rbXTr2qomOK2sBXUc41QBv34fHulaP7UU/OOY3aYXv3J+fOnXPHjo4WP+c4cOCAOzanxXf0nMRrA5/zXDlnrBQ/b/UsLS2Vikl5re3Pnz9femwkaiHvtYiP8lpZWSmVUy94JRAAAAAAAEANsAgEAAAAAABQAywCAQAAAAAA1ACLQAAAAAAAADXAIhAAAAAAAEANsAgEAAAAAABQAywCAQAAAAAA1MBIzmAze1DSgqQNSesppVuiMSmlUjEzizZcarvdbLvRKF4rW2mtuGOXV4rjFy9edMeOjPinp+HEW62WO9aLR/PhxcPzFEhOXtF59HjnUIrna319vTCWcx4bwXyNjo668ei4PGUfi1VVph4NQvS4zRFdpx7vGh5kzlW1vLw87BQqZWJiwo0zX92rSi2ampqq5LYXFhYGst2dypuPOpqdnXXjc3Nz25TJztdLLUopaWNjo3Bbg7qH8PYpSc1m042PjY0Vxh555BF37Be/+MXC2EMPPeSOjWrV+Ph4YWzFeW4YxaO59uYr917Puwai8+jx5kqSVldX3bhXQ3POY5TX3r173Xg03uPNZ85cb9aPO/9/mlI63YftAEAu6hGAKqAWAagCahGAJ+DtYAAAAAAAADWQuwiUJL3fzD5uZi/tR0IAUBL1CEAVUIsAVAG1CMCWct8O9k0ppTkzm5H0ATP7h5TShzf/QKfovFSSrrv22szdAUAhtx5trkXXX3/9sHIEcOXruhZ5n2cBAJm6rkVPetKThpUjgCHIeiVQSmmu89+Tkv5Q0rO3+Jk3ppRuSSndsn///pzdAUChqB5trkUHDx4cRooAaqCXWlTHD2UHsD16qUXRh9wCuLKUXgQysz1mNnXpa0nfKenT/UoMALpFPQJQBdQiAFVALQLgyfkT1DWS/rDTEnxE0u+nlP68L1kBQG+oRwCqgFoEoAqoRQAKlV4ESil9XtLX9DTITM1m09umE2v5mw7264f9uJfX0oUld+zFpeL42tqaO9abqygvLyZJrVZx3Mwf681XMJUKzpQrOiZPq+VfPxsbG6Xj0ba9eHRE0bUZxeuiVD0qqdHwX0CZ87jMsbKy4sbHx8cHtu8cJ0+eLIzNzMyU3u709LQbn5+fL73tK9Hy8vKwU7gibGctikxNTRXGFhYWBrbfQ4cOufHFxcXC2OTkZOmxuDLMzc0NO4UrQq+1qNFoaPfu3YXx9fX1UjEpfj6TM9a7P//CF77gjj1+/Hhh7Pz58+5Yb66ivKLnHKurq4WxaK69txhH+80R5ZUjugYuXLhQGMu5N869rnOu++1Ai3gAAAAAAIAaYBEIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBkq3iB+EnPbJXtvmVmZLPK8N99q63+Z9faO4vVzUWnyw7b+9NtbRSKe9fIpyHlyLbH+vQdv7lp+3d23mnKfc2cja94CO6UoXPW4H2QbeE7WAd2tkcEyDNKg28FVtAV/V1vUnT5504znnCcMxyDbwnqiNu5dX1CK+qry25rOzs9uYSfeiVuzDyju6bqemprYpE2yW017cuz/x2qF3w2vDfe7cOXes9/s2ai0+rPbf0f1aTnv5Ycltxe6NHxkpv9SRc81LedeIt+9+XXu8EggAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAamBkW/eWpJTSYDbtbLfVarljWy0/p5GR4rWyRsNfRzOzwlg0F1HeOWO9uJezJMmb68zTG+7bkXNtNceabtw7z9E14OWVgvMUHVPOfKF6FhcX3fjk5OQ2ZYJBmZ+fH3YKW5qZmRl2CsCONTs7O+wUelbVnKempoadQi2llLS+vj6QbXvbXVlZcceurq66ce96GR8fd8eOjBQ/Dd7Y2HDHRnl78eiYvHiz6T9fKbvdbuTsO5pPz549e9z4xMREYWxsbMwd612bUc5R3Lu+qoBXAgEAAAAAANQAi0AAAAAAAAA1wCIQAAAAAABADbAIBAAAAAAAUAMsAgEAAAAAANQAi0AAAAAAAAA1UKneZQNrDx62Dvfj3rajnJPTMz2njXuUl7dfSWql4m1HLc/L5tRN3Mzbd/kW8Lmt1L3xFsxXy2khmNPWvr1zWsTvJNF1OMgW8FE9GZSojWrUwtUzyHbrJ0+eLIxVtZ26l7M02Lx34nzVWdSGe2FhYWj7HtZ+B3nMdTM3N+fGB9me3juPtJ8vb1jtwSNei+8o50G2rveOOaf9fM49k3e8UpxXzjXgyW2l7uUVbdub67W1tdI57QS8EggAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAamAk+gEze4uk50k6mVJ6Rud7+yS9U9INkh6U9MKU0rl4W1KjUbzulFLqKumeWRA2/weieNmxuftttVqFsWgmvW1H52GQx5TccPnzEF9ZwTE7+24Ex7QxqOtaUnKugStRP+vRMOQ8pqtsYmJi2Cn03ZEjR4adwpZWV1cLYzMzM9uYSXX2PQw7vRYtLCwMO4WBmJqaKozt1GM+duzYUPZ79OhRN+7N9ezsbL/T6ZqX15WoX7Wo0WhofHy8ML6xsdGnjHvTbDbd+MhI+FS21Nhou97zWcn/Xb2+vu6O9Y45GptzTFE82ndZudv18h4bG3PHLi0tZe3bM6j56pduXgn0Vknffdn3Xinpgymlp0r6YOf/AWDQ3irqEYDhe6uoRQCG762iFgHoUbgIlFL6sKSzl337+ZLu6Hx9h6QX9DkvAHgC6hGAKqAWAagCahGAMsp+JtA1KaUTktT5b71eBw6gSqhHAKqAWgSgCqhFAFwD/2BoM3upmd1lZnedPnNm0LsDgC1trkWnTp0adjoAampzLar6ZwYAuHJtrkVnz17+YiIAV7Kyi0CPmNkhSer892TRD6aU3phSuiWldMuB/ftL7g4ACnVVjzbXooMHD25rggBqoedalPOhpgBQoOdatG/fvm1NEMBwlV0Eeq+kl3S+fomkP+5POgDQM+oRgCqgFgGoAmoRAFe4CGRm75D0t5JuNLOHzezHJL1O0neY2WclfUfn/wFgoKhHAKqAWgSgCqhFAMoIX4ecUvrBgtC39bqz1N5er8MeG1tWtMuyOeWzIe03kzddmYdkGafCzNl5kFcr2m9qFYeC68fLKxobxRuNgX+sV6X0sx6VFX2Gh/f2jlar+DoapsXFRTc+OTm5TZn0ZmVlpTA2Pj6ete35+fms8UWmp6fd+AMPPODGl5eX+5kOSqpCLVpYWHDjU1NT25QJvLmOzlPk6NGjWeOLzM3NZcVnZ2f7mQ5K6lctSilpY2OjVA45n20W7bNsTui/nLcvN5vN0mO9e70oHl0/Xl7RPXu07dz70EGr1zNIAAAAAACAmmIRCAAAAAAAoAZYBAIAAAAAAKgBFoEAAAAAAABqgEUgAAAAAACAGmARCAAAAAAAoAbK93rbZlHn8WE1ec/jZx01kx7WMSdnz5bZI95t854xNtxu1Krd6SGf0/Y7bBEfbLth5ddxo31jazktMqsqagEftZAflqq339xKbut5rxXqIOdjYmLCjT/66KOFsZ14nnaCOraAj9qWD0tuG/hhyG3x7l1/g5yPnOt+J56nnSC6L8ppIV9Va2trbnxYx+ztN/f+NafNuze20ch7TYp3zFF7eU/UAj66BsbGxkrvO9p2P/BKIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACogZFhJ3ClS17MLGvb3mhvv1kblmTRD1SQZc51cma01fJnu9Eo3neYVxDfaG34491Ne3mV3iwqan19vTC2vLycte2JiYmBbHt6etqNz8/Pl972TjU+Pl4YO3nypDt2Zmam9H6j8+jllSO6BrDzLCwsFMbm5uaytj01NVVqv5Eor9nZ2dLb3qly5nMn7hdXHu++yIt1Y2Sk+Cl27rbL7reqcnNutVqFsdXVVXfs2NhYYazZbJbOSZIuXrxYeuzo6GhhLDevS3glEAAAAAAAQA2wCAQAAAD7K0zHAAAgAElEQVQAAFADLAIBAAAAAADUAItAAAAAAAAANcAiEAAAAAAAQA2wCAQAAAAAAFADO6aP3EBbnme06Q7bvGe04Y7ySo3iNTxLWTPmym23Pqj9+i3Pw773Lm86UypuTdhW3MovavPXcM6xJG1sRPsuNqzzuNNFrT29VpfR+fTaXEb7HmYr9oceeqgwltOWvI4t4HNaog9zvrz29IO8BurcQj5qlZ3TLt0bG42P2qlH285x//33F8Zy2rjXsQV8dB4XFxcLY0ePHu13OqiwQbY8z2mHvbGxUXps1LY8ysvb9yDbuPerfXi/9+vd/+bOh9eKPbo2vRbxu3fvdsdOTEy48aWlJTfu2Y7zyCuBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGRqIfMLO3SHqepJMppWd0vvcaST8u6VTnx16VUnpfuK322MJ4q9UqjKWU3G03rHg9y1S8T0nytxzHXc7xyslZktTw8zYnsaxjCuY6OaOjuY5414ei8+iEW6n42pKkhjXdeNn9StKGt+/gHEfx9dZ68dBG+TVe77E4LP2sRTlGRvyyubKyUhiL5nV9vfh8StLy8rIb95w8ebIw9sADD7hjp6enS+/3RS96kRufmZkpjHk5R+68887SYyPRfB05cmQo2845T4M0Pz8/7BT6rgr1aGpqKivuWVhYyIp75ubmSo9dXFwc2H5nZ2dLj/Xk5ByZnJwsve9obI5jx44NbNs5Dh06NOwU+q5ftcjM1GwW3wt79zYbGxtujuPj44Uxb5/dbDu6bxqW6LjKiuYjiufIOSbv/je6t52YmCi932g+lpaWSm874v2eHBsbK71d77HYi26eJb5V0ndv8f03pJRu7vwb6JMuABC1CEB1vFXUIwDD91ZRiwD0KFwESil9WNLZbcgFAApRiwBUBfUIQBVQiwCUkfOZQC8zs0+Z2VvM7Oq+ZQQAvaEWAagK6hGAKqAWAShUdhHoNyR9paSbJZ2Q9PqiHzSzl5rZXWZ21+kzZ0ruDgC2VKoWnTp1qujHAKCsrurR5lpU1c+zALCj9VyLzp7lxURAnZRaBEopPZJS2kgptSS9SdKznZ99Y0rplpTSLQf27y+bJwA8QdladPDgwe1LEkAtdFuPNtei6IPmAaBXZWrRvn37tjdJAENVahHIzDZ/vP73S/p0f9IBgO5RiwBUBfUIQBVQiwBEumkR/w5Jz5V0wMwelvSzkp5rZjer3Wn8QUn/qus9+r3Ju97ME3jtwYOh4V69tuUZLb6T1+Nd0nrQ1rzptAC3qP2806ovmi/viJPb4j1qAS93rqPzlJVX058vr4V8WnOHamW1uJXf0kW/NeHF5YtuPKXiWQlbxGc83Iahn7UopeS2WNy1a1dhLGrz7rVCzW0V7LVMj9qS33bbbYWxe++91x37ile8wo17c+nNRzT28OHDpcdGOef4rd/6LTd+6623Fsai+bj99ttLbzuHd21148477yyMRa3rH3jggcLYzMxM6ZwGqV/1qNFouK3cc1qxe2NzWp5H247qXE5r8mc961lu3MvLm+do7I033jiw/Q7SoOZjkHKvTe/6O3HihDvWuzaHeR49fX+ets1WV1fd+CDfMpvzSsyo9fjo6Gjp/UZzUlbU4j163jCoV65G2929e3fpbZ87d86Nn3E+quahhx5yx0a1yrtGclrE90t4NlNKP7jFt988gFwAoBC1CEBVUI8AVAG1CEAZOd3BAAAAAAAAsEOwCAQAAAAAAFADLAIBAAAAAADUAItAAAAAAAAANcAiEAAAAAAAQA2wCAQAAAAAAFADYYv4fkpJSq1WqbEmK73fcKQFP+GEW0qlx6Zgv63kb7vRKD8nLfeYcvg5Rxl78ZzzGJ3ijWCu3Tlp+Gup3rY3gsdDcHWpOVr8EI4eM95jcWMj7yrYCZrNZmEsOedseXnZ3e74+HhhbGJiwh370EMPufEjR44Uxm699VZ37KlTpwpjhw8fdseOjPi/KtbX10vFJP88RLy5HqTomHLy2rNnT+mxOWZmZrLGe9ffwYMH3bHedX2lazabmpycHMi2FxYWCmNTU1Pu2Lm5OTe+uLhYGDt06JA71tt3lFe07aNHj7rxsqK8hsU7x1Je3seOHSs9dlDXtJR3zNFY77r2YleClJI2NjZKjc35PZ4zNpe37yivaK5yHnvRPcZO5N1H5s61J7of87a9urpaer+StHfv3tJjvX0vLS2V3u5mvBIIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGpgZHt3l5RSKg6bObHyezXLW+vaaLUKY2tra+7YtY314u2m4u1K0kgjOOhG8XE5s9yOO5tOzvFKCs5FkHMQjvIuu2nv0pKkaKq9DTSazWBo8djwPHmPlyieM7bmlpeXC2Pj4+Olt/voo4+WHitJt912W2Hs1KlTpbe7vl5cpySp4dQaSRobGys9Ntp32bFeTpK0urpaer+DFM1HdFyellPbo+t6ZWWl9H5f/epXlx6LYgsLC0Pb9+TkZOmxs7OzhbFDhw65Y2+88UY37s1JtO2pqSk3PqixgzTIayRn297YaC5z9ru4uFh67JUupZT1+7isZnAPHfF+l58/f94de+HChcLYxsaGOzb6XZxzr+jNSfS7eGSk/FP73HNRVpRzlJcX3717d9a2PdHjxbuGoutrOx6LvBIIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBlgEAgAAAAAAqIFtbRFvMrddcHIaZuc0s24E/b+bTX8tLDktvr328ZLU8hIPWic3RoLW4yPO+GjC3MT8+fJannuxbuJe2/KwXbq7Y3doGG8410gz+edpZGy0MDY67reYHA1aUI40ix/CreRfm3Ku3aYNp01kVUxMTBTGouvQc9VVV7nxgwcPunGvRXzUjtRrZxq1ovRai0fjozaqOfPptfaM2m8OshVqzjFFeQ9KdA1E19f09HRh7NZbb3XHnjx5sjB25513umNRPV4LeMlv1R61gI/avB89etSNe6K86+bYsWMD23bOXEct4ufm5gpj0fXjbftKby/faDTcOu/9bsppZx39bolafHuidurevU107zI1NVU6Hs2X1/Y+4rVbz2m1LuW1PM/ZbxT3rpEor/379xfGrrnmGnfsgQMHSueVcw1457gXvBIIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGogbDRvZoclvU3SkyS1JL0xpfQrZrZP0jsl3SDpQUkvTCmd87aVTGqZu7fust5CS6kw1hhpumMtiBdvOR7bHCue4pE06o4dGfXj3r5Tq+WO9Y7KgtNgVrx2aA1/sEXnODl5ObFobMQaGeuh0TE7244yTtFPePv2H2zxia6YftaiyNraWumxY2Njpcfeeuutbnxpaan0tldWVgpjFlwLjeDxkXPMzWZxHVtdXS29342NjdL7jbSC+poyalGU16C2PTLi3w54148kzc/PF8bGx8fdsTMzM4Wx2267zR376le/2o0PQj9r0fLyso4dOzaQPBcWFkrFJGlqasqNz87OFsaOHj3qjn3Ws55Ver9R3Nv2sEQ5R+ciZ+zi4mLpbeeci8nJSXesl1fuNeBdm4OUM9dl9bMWtVqtsM6X5W03un/YvXt36f1G18qBAwcKY9E9xNVXX1163znzHN0jePHod3HEm5P19fXSYyM5eUf3kblz4sm5N86Zr25188x3XdJPpZSeJukbJP1bM/tqSa+U9MGU0lMlfbDz/wAwKNQiAFVALQJQBdQiAKWEi0AppRMppU90vl6QdJ+kayU9X9IdnR+7Q9ILBpUkAFCLAFQBtQhAFVCLAJTV03tgzOwGSc+U9FFJ16SUTkjtIiSp+PXcANBH1CIAVUAtAlAF1CIAveh6EcjMJiW9W9LLU0rFb/5/4riXmtldZnbXmTNnyuQIAI/pRy06ffr04BIEUAv9qEXb8b5/AFe2ftSic+eyPkoRwA7T1SKQmY2qXVzenlJ6T+fbj5jZoU78kKSTW41NKb0xpXRLSumW/fv39yNnADXVr1rkfRggAET6VYtyPqAcAPpVi6IPOgZwZQkXgazdNubNku5LKf3yptB7Jb2k8/VLJP1x/9MDgDZqEYAqoBYBqAJqEYCywhbxkr5J0osl/b2ZfbLzvVdJep2kd5nZj0l6SNK/6GaHXnPbnNa3bqvsoIV3K9jvqtP27uKq3+bPayTcGPWn35r+Gl3DiUdjvXjUlrzZKP7LZdRKOpI2imcsanvvXT9RC+xmEG95ebkj/X1HY/0jlkZGRgtja+t+m3P3uk/Rnoeib7UopaTl5eWBJVokagEftarMeXx5rziIWp7ntESPHnue6FUSXl7RfqNjypnrnGOO2qxGrdzLio43ujanp6cLY0tLS+5Y75hOnTrljh2SvtWiVquV1SLcc+LEicLYoUOH3LE5bbqjNu057eWjNtxeXtExeXJankeivO6///6B7DcSnYuc69bLO7fF+8c//vHCWM51PTc3544dRot49fk5mvf21EG9dTX63RLt17sOc35/7N27143v2rXLjXut7aO291Hc481nTstyyf9dHrVi9+5tovua6F4wusfI2XYOr55E9dObr349FsO7yZTSRyQV3dF+W1+yAIAAtQhAFVCLAFQBtQhAWXkv2wAAAAAAAMCOwCIQAAAAAABADbAIBAAAAAAAUAMsAgEAAAAAANQAi0AAAAAAAAA1wCIQAAAAAABADYQt4vsqSSml4rATizSs/HrWxsaGG19bXS0Vk6SxsdHC2ERzwh87Pu7GJ3bvKoyNjBbvV5IajeL58mLD1ArO08baemFsfb04JkmtdX/bqysrxWNb/tiWcw1Yo6izZ1tq+Y+JjdRy4x5vz+UfiRikVqv4fEf1c9WpVc1m0x1r5l+ngzKs/Up5c52T99jYmBv38orq3MhI8a/86JjGg99Hy8vLbtwT5Y3qWVhYKIwtLi6W3u6JEyfc+NTUlBufm5srPbaqcvKO5tPjneMoHuV86NChUjlF+5Wk2dnZ0mOj+JUspeQ+H1pbWyu97YkJ//mOZ2lpyY2fPn26MHb27Fl37P79+wtju3fvdsfOzMy48ac85SmFsb1797pjvd+30e/iYYnO07lz5wpjuY/LRx55pDDm3ftK0oEDBwpjo8Fz6Wjb0frCsFXz2T4AAAAAAAD6ikUgAAAAAACAGmARCAAAAAAAoAZYBAIAAAAAAKgBFoEAAAAAAABqgEUgAAAAAACAGmARCAAAAAAAoAZGtnuH3qpTMiuMtZTc7bZarcKYmb/WlZyxkpRS8b73Tl/l51V8SLKmn1ej2XTjuyZ2FY8d9cfKmWtzYpLUcObTGv7YaNteNLX8a8A7jxsbG+7Y9dVVN76yvFIYm5iY8Le9vl4Ym9wz6Y5tBPO55uTtPSYkSc517e9152s0GpqcLJ5775yNjY0NIiVJ0spK8XUmSSMjxSU7emx5Y70a1w1v295cSnHenkajuBaF13/Am5NmUJtz9r0a1CJvrr1YJDqmpaUlNx7VQWxtdHRUs7OzhfGFhYXC2Nzc3CBSCvcrSVNTU4Wx+++/3x3r1V5vLrrh7fvGG290x3p5RRYXFwey3Wjb0XmK4jljvbxyRNd1tN+cY64zM3Pvb7zfEcvLy+62vXj0uye6L/Lu72+66SZ3rLfv3bt3u2N37Sp+DiZJ1157bWFs79697ljvd3k0X158fHy89NhIdJ68e5vo/uL06dNu3KsZ3nmQpEcffbQw9tSnPtUdGz0f8PKO7vWi5639wCuBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBrY3hbx5rcD9lryWti02ml5HnQ/DjqPu22Ip6en3bHJSztqjRy0Bx8ZHS297ZSKWxhH3aLdQ4omU368ldGq2ru2Rr25kjQ+6rf52zVR3Aoyav/qXdcjQTvG0YYfd1sIDmgu625+ft6Nnzx5sjB25MgRd+xtt93mxqM2mh63vob1wr+WvOswajnqtcmM2m8Okjcnue3nPdExD2rfg2xHur6+PrBt15nXpj1y4sQJN37o0KHS287Zd9T+O8rLm5NBtlP3RC3Pr0Q57eNzrusI7eOLmZnbmtz7HRHdY3ui3z1R3PudmdMiPrp3idqte23go217x5zzuzpq4x7J2bd3zN5cdRP32sAfPXrUHevdU+3evdsdG10DFy9eLIzl3Bd5j9Ne8EogAAAAAACAGmARCAAAAAAAoAZYBAIAAAAAAKgBFoEAAAAAAABqgEUgAAAAAACAGmARCAAAAAAAoAZYBAIAAAAAAKiBsNG8mR2W9DZJT5LUkvTGlNKvmNlrJP24pFOdH31VSul9OclYzmBPSln7bVrxTyQL1tG8jTcyxkpqrW8UD234g83ZdyMYm5z5TC1/rlNq+Xk5c+3lLPnTlVr+flvBZHvRkWbTHduMrhGHN9eS1HKOq+HMpRSsAPu7HYrtrEUjI8WlcWJiwh07MzOTs2vXhQsXCmN79uxxxy4vLxfGxsbG3LEbG8W1RpIazmMzGuvNtXd9R5rB43J9fb30tr06JcWP2xw523bra3BMg3T8+PHC2J133rmNmXRnO2vR1NRUzvDSFhYW3Pjc3FzpbZ84caIwdvToUXdslNehQ4cKY9FcDmquo5xzRDkPct85277//vv7mEn/eNf14uLiNmbSne2sRdHv1GHx7j+inL34+Ph46bGS//iItu3dk+Xcr62urrpjo/si734tyssT5ZUjujf2jikSzZd3XDmPp+i+ulvdHPm6pJ9KKX3CzKYkfdzMPtCJvSGl9J/6kgkA+KhFAKqAWgSgCqhFAEoJF4FSSickneh8vWBm90m6dtCJAcBm1CIAVUAtAlAF1CIAZfX0PhUzu0HSMyV9tPOtl5nZp8zsLWZ2dZ9zA4AtUYsAVAG1CEAVUIsA9KLrRSAzm5T0bkkvTynNS/oNSV8p6Wa1V6FfXzDupWZ2l5nddebMmT6kDKDO+lGLTp06tdWPAEDX+lGLcj6bCgCk/tSic+fObVu+AIavq0UgMxtVu7i8PaX0HklKKT2SUtpI7U/6fZOkZ281NqX0xpTSLSmlW/bv39+vvAHUUL9q0cGDB7cvaQBXnH7VopwPpQSAftWiq6/mxUJAnYSLQNZuGfJmSfellH550/c3t2D4fkmf7n96ANBGLQJQBdQiAFVALQJQVjd/gvomSS+W9Pdm9snO914l6QfN7Ga1m0k/KOlfhVtKg2ud67USjtqwWdSG22nFvrx80R076rTMGwna6UUte/0W30ELY6eVe04b96gtuax8S7xG0GrdaxMZXnVBK+q0URzfiMZa8d6j+bJgnTap+No05xy3f8BpF90YXrtoR/9q0ZDMz8+78ZMnT7rxI0eOFMZuvfVWd6zXkjR6NULUqj2n/ua0gfdELTQH2RLdrZFOnZIGm7fXwjU6D9PT027ce4vl7bff7ie28+z4WpTbWtxr8567b4/XAl7yW3xH+52dnS2V0zBF58E7j9FcRtv22rxHc+21W4+uvZx4znVbUX2rRSmlsOV1WcvLy4WxqIV39DvRu5a8eiBJBw4cKIzt27fPHZvT4jtnnnPauEc55xxTNNa7B42srKy48YsXi5+LR+3nvesrZz4i3mNCkkZHRwtj3r1cL7rpDvYRSVvddb6vLxkAQBeoRQCqgFoEoAqoRQDK6qk7GAAAAAAAAHYmFoEAAAAAAABqgEUgAAAAAACAGmARCAAAAAAAoAZYBAIAAAAAAKgBFoEAAAAAAABqIGwR319JKaXCqNlWXQ4vDS0e11U8g5fz2uqaO7ZhxetsIyP+9Dcbfjx58xVIreJjajkxSXIOSY0gp4yUZVFe3rWlYGyUd6NZGBvxJkTSyupqYcy7tiSpGWzbu4a8cyxt3VP0sdjgHk47XvS4nZiYKIwtLy/3O53HHDx4sPTY+fn5rH2PjY1ljS9r1XlsRTktLS31O53H7N69e2D7ja4/T6vVKoxduHDBHfsLv/ALpff7wAMPuPGZmZnS256eni49dqebmpoqPXZhYaGPmfS27Zx9z83NufHZ2dnC2IkTJ9yxx44dK5WTJB06dKj0fneq6Fx4vGs32m7OfE5OTrrxnMfUlXCeNzY2CmPNZvF98DCtr68Xxk6fPu2O9Y4puha8e71o25GVlZXCWHQf6eUV5ZRzf+Hdj0n+tRWJ8vbu96Kx3jUS5Rxt27uGorFePOc8bcYrgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBlgEAgAAAAAAqAEWgQAAAAAAAGqARSAAAAAAAIAaYBEIAAAAAACgBvrTaL5bZmo0itedUkqlYp0fKBfrIm5OeHxszB27urJSGFu+eNEdq4a54T179hTnNT7ujh1pNgtjyTlHUnCeWhvuWJl/TO5+g3grrRfHWi137MiI/1Dwjnltdc0de2FxsTDmPR4kadfEhBsfHys+z0n+MSdvToL5QjHvWpoIzufMzIwbn5+fL4xNT0/7iWW49dZb3fiKU+eGZZg5LSwsDGzbGxvFNfbgwYPu2Fe/+tX9TucxDzzwQGEsuq69x0VUm1FsampqKNs+ceJE6e0eO3bMjR86dMiN33///aX3nWNY+62qqAYuOvdFkcnJydJjo8eEF4/G5lz3VWBm7nOH9fXie2zv99Kgeb8jDhw44I595JFHCmPHjx93x44Fz/+OHDlSGIt+J3rP76L7SO9cLC8vu2NHR0fdeI61teLnSl5Mih973jGfOnXKHev9zonO8fXXX+/Gr7nmmsJY9JjZjscUrwQCAAAAAACoARaBAAAAAAAAaoBFIAAAAAAAgBpgEQgAAAAAAKAGWAQCAAAAAACoARaBAAAAAAAAamB7e68mv9W2Oe3Do1baXgvwqCl5I2hb7rUf3BWMbXmt+px2i5LUWvfbdK+PFrfUG236p3akUdwi3oIZa3nnMBjbSH7cbwMfRJ2wBfttrfut+FZXVwtjFy9e9PNyrs2RUb/94EhwHje8torehMh/vEXnEeVE7a6j1p9ee0+vfbyU10L+9ttvLz22qqL5OnnyZGHMa/2au9+c85TDa/EuDa/N+8rKSumxKJbbPt5rAR61cffGRq3Dd3ob7q3kzFeOYbY8r2qb95z97nQpJbcNfE4d9+Ru17uWohbe3u+Xubk5d2z0uLz66qsLY7t373bHRveCnqjduqfZLH5uKA2ubXm032iuT58+XRg7fvy4O9a75vfv3++OjWrC0tKSG/d4j4tovrrFK4EAAAAAAABqgEUgAAAAAACAGmARCAAAAAAAoAZYBAIAAAAAAKgBFoEAAAAAAABqgEUgAAAAAACAGggXgcxswsz+zszuMbN7zey1ne8/2cw+amafNbN3mpnf6xoAMlCLAFQBtQhAVVCPAJRR3IT+y1YkfWtKadHMRiV9xMz+TNJPSnpDSulOM/tNST8m6Tf8TSWplYrDTSsMmRXH2lsu3m6r1XLHNhr+Wthos+nGPVOTU4WxyT2T7tiUnLmSf1xN84/J/E27GnLOUzA2Oo+tVHxMwXTInL03mv58bGxsuPGxkdHi2FRxTPKPOTrH0Xwm5xqIrmt329FkD0ffalFKSevr64XxkZFuSuPWvHmPalG038nJ4pqxuLjojp2fn3fjV5qJiYms8UeOHCmMDXIuq3qeovnMecx4NbKZ8ft3gPpWi0ZHR3Xo0KHC+IkTJ0onubCwUBibmiq+N8mNz83NlR4b7Xcn2qnH5F2Xw5R77ZYd6/3+HbI+Pk8rJ6r/Fy5cKIytrKy4Y8fHx934vn37CmNnz551xz796U8vjH3VV32VOzZ63rC8vFwY2717tzs25/ee97s62m50Hr1zFc3H6Gjxc6Uor6WlJTd+4MCBUrFo39ExRXl746PrejvufcJXAqW2S88uRjv/kqRvlfRfOt+/Q9ILBpIhAIhaBKAaqEUAqoJ6BKCMrj4TyMyaZvZJSSclfUDS5ySdTyld+lP6w5KuHUyKANBGLQJQBdQiAFVBPQLQq64WgVJKGymlmyVdJ+nZkp621Y9tNdbMXmpmd5nZXWfO+C/LAwBPv2rR6dOnB5kmgCtcv2rR6urqINMEUANl69HmWnTu3LlBpwmgQnrqDpZSOi/pryV9g6S9ZnbpzYPXSdryTeAppTemlG5JKd2yf3/x+zYBoFu5tSh6jzAAdCO3Fo2N8VmtAPqj13q0uRZdffXV25cogKHrpjvYQTPb2/l6l6Rvl3SfpL+S9AOdH3uJpD8eVJIAQC0CUAXUIgBVQT0CUEY37TwOSbrDzJpqLxq9K6X0p2b2GUl3mtnPS7pb0psHmCcAUIsAVAG1CEBVUI8A9CxcBEopfUrSM7f4/ufVft9pD0xqlG+X7Wk2yrdSi/brxZtR63qvPbif1kB5raqjl4dFbcs9OS3Rc/YbtTyP2qnnXJve2EZw/YTxIG+Pe0RZkz0Y/axFZpbV0toTtYEflKh97fr6emHMa2U6TMNsyTus87i4uBj/0AAMsgV8xKuRg9xvWf2sRWtra1lt4D3Dak0+Ozvrxr3W9V5smKJjGqRhnce5uS3fzThwg2wBH9mJ12Z/n6cV8+4hInv27BnYfr14tF+vhXfUHjySc516nxUXvYU4p7V4NNfetgfZ0jxqp762tlZ62969ntfWXorvm6K8PbnXXzfKP4MEAAAAAADAjsEiEAAAAAAAQA2wCAQAAAAAAFADLAIBAAAAAADUAItAAAAAAAAANcAiEAAAAAAAQA2wCAQAAAAAAFADllLavrCybIEAAAYhSURBVJ2ZnZL0j5u+dUDS6W1LoHvk1b0q5iSRV696zesrUkoHB5XMoFGLspFXb6qYVxVzkqhFV8p52S7k1Rvy6h61qHrnRCKvXpFXb66EvLqqRdu6CPSEnZvdlVK6ZWgJFCCv7lUxJ4m8elXVvLZLVY+fvHpDXt2rYk5SdfPaLlU9fvLqDXn1pop5VTGn7VTV4yev3pBXb+qUF28HAwAAAAAAqAEWgQAAAAAAAGpg2ItAbxzy/ouQV/eqmJNEXr2qal7bparHT169Ia/uVTEnqbp5bZeqHj959Ya8elPFvKqY03aq6vGTV2/Iqze1yWuonwkEAAAAAACA7THsVwIBAAAAAABgGwxlEcjMvtvM7jezB8zslcPIYStm9qCZ/b2ZfdLM7hpiHm8xs5Nm9ulN39tnZh8ws892/nt1RfJ6jZl9sTNnnzSz7x1CXofN7K/M7D4zu9fM/s/O94c6Z05eQ50zM5sws78zs3s6eb228/0nm9lHO/P1TjMb2868hoFaFOZBLeotL2pRb3lRizahHoV5VK4eUYv6lhe1qEKoRWEelatFTl7DfmxRi3rLa/tqUUppW/9Jakr6nKSnSBqTdI+kr97uPApye1DSgQrk8RxJXyvp05u+90uSXtn5+pWSfrEieb1G0k8Peb4OSfraztdTko5J+uphz5mT11DnTJJJmux8PSrpo5K+QdK7JL2o8/3flPQTwzyv2zAP1KI4D2pRb3lRi3rLi1r05bmgHsV5VK4eUYv6lhe1qCL/qEVd5VG5WuTkNezHFrWot7y2rRYN45VAz5b0QErp8ymlVUl3Snr+EPKorJTShyWdvezbz5d0R+frOyS9YFuTUmFeQ5dSOpFS+kTn6wVJ90m6VkOeMyevoUpti53/He38S5K+VdJ/6Xx/KNfYNqMWBahFvaEW9YZa9DjUo0AV6xG1qG95DRW16HGoRYEq1iKpmvWIWtSb7axFw1gEulbS8U3//7AqMOkdSdL7zezjZvbSYSdzmWtSSiek9oUraWbI+Wz2MjP7VOdliNv+8sfNzOwGSc9Ue+W0MnN2WV7SkOfMzJpm9klJJyV9QO2/+pxPKa13fqRKj8tBoRaVU5nH1RaoRb3lJVGLqoJ6VE5lHluXoRb1lpdELaoKalE5lXlsbaES9Yha1HU+21KLhrEIZFt8ryotyr4ppfS1kr5H0r81s+cMO6Ed4DckfaWkmyWdkPT6YSViZpOS3i3p5Sml+WHlcbkt8hr6nKWUNlJKN0u6Tu2/+jxtqx/b3qy2HbXoyjL0x9Ul1KLuUYseQz26cgz9cXUJtah71KLHUIuuLEN/bEnUol5sVy0axiLQw5IOb/r/6yTNDSGPJ0gpzXX+e1LSH6o98VXxiJkdkqTOf08OOR9JUkrpkc7F2pL0Jg1pzsxsVO0H8dtTSu/pfHvoc7ZVXlWZs04u5yX9tdrvN91rZiOdUGUelwNELSpn6I+rrVTlcUUtKqfmtUiiHpU19MfW5aryuKIWlUMtohaVNPTH1laq8NiiFpUz6Fo0jEWgj0l6audTrsckvUjSe4eQx+OY2R4zm7r0taTvlPRpf9S2eq+kl3S+fomkPx5iLo+59ADu+H4NYc7MzCS9WdJ9KaVf3hQa6pwV5TXsOTOzg2a2t/P1LknfrvZ7Yf9K0g90fqwy19gAUYvKoRYV50At6i0vatGXUY/KqVw9GvbjqpMDtai3vKhFX0YtKqdytUiqxGOLWtRbXttXi9JwPvn6e9X+FO7PSfr3w8hhi5yeovYn4N8j6d5h5iXpHWq/BG1N7RX5H5O0X9IHJX228999FcnrdyX9vaRPqf2APjSEvL5Z7ZfFfUrSJzv/vnfYc+bkNdQ5k3STpLs7+/+0pJ/pfP8pkv5O0gOS/kDS+HafyyFcO9QiPxdqUW95UYt6y4ta9Pj5oB75uVSuHlGL+pYXtahC/6hFYS6Vq0VOXsN+bFGLestr22qRdTYMAAAAAACAK9gw3g4GAAAAAACAbcYiEAAAAAAAQA2wCAQAAAAAAFADLAIBAAAAAADUAItAAAAAAAAANcAiEAAAAAAAQA2wCAQAAAAAAFADLAIBAAAAAADUwP8PZ3oSZf94b0gAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f29b1d42588>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Convert and image to HSV colorspace\n",
    "# Visualize the individual color channels\n",
    "\n",
    "image_num = 966\n",
    "test_im = STANDARDIZED_LIST[image_num][0]\n",
    "test_label = STANDARDIZED_LIST[image_num][1]\n",
    "\n",
    "# Convert to HSV\n",
    "hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)\n",
    "\n",
    "# Print image label\n",
    "print('Label [red, yellow, green]: ' + str(test_label))\n",
    "\n",
    "# HSV channels\n",
    "h = hsv[:,:,0]\n",
    "s = hsv[:,:,1]\n",
    "v = hsv[:,:,2]     \n",
    "        \n",
    "# Plot the original image and the three channels\n",
    "f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))\n",
    "ax1.set_title('Standardized image')\n",
    "ax1.imshow(test_im)\n",
    "ax2.set_title('H channel')\n",
    "ax2.imshow(h, cmap='gray')\n",
    "ax3.set_title('S channel')\n",
    "ax3.imshow(s, cmap='gray')\n",
    "ax4.set_title('V channel')\n",
    "ax4.imshow(v, cmap='gray')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='task7'></a>\n",
    "### (IMPLEMENTATION): Create a brightness feature that uses HSV color space\n",
    "\n",
    "Write a function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. The only requirement is that this function should apply an HSV colorspace transformation, the rest is up to you. \n",
    "\n",
    "From this feature, you should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light. You may also define helper functions if they simplify your code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value\n",
    "## This feature should use HSV colorspace values\n",
    "\n",
    "# Function to calculate brightness of an image using HSV color space\n",
    "def norm_region_brightness(rgb_image):  \n",
    "    \n",
    "    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)\n",
    "    v = hsv[:,:,2]\n",
    "\n",
    "    top_bottom = 1\n",
    "    left_right = 9\n",
    "\n",
    "    v_crop = np.copy(v)\n",
    "    v_crop = v[top_bottom:-top_bottom, left_right:-left_right]\n",
    "#     print(\"Image dimensions: \", v.shape)\n",
    "#     print(\"Cropped image dimensions: \", v_crop.shape)\n",
    "\n",
    "#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))\n",
    "#     ax1.set_title = (\"Original V channel\")\n",
    "#     ax1.imshow(v, cmap=\"gray\")\n",
    "#     ax2.set_title = (\"Cropped\")\n",
    "#     ax2.imshow(v_crop, cmap=\"gray\")\n",
    "\n",
    "    # Set regions for each of the 3 colors: [rows, columns]\n",
    "    red_region = v_crop[4:10, 2:12]\n",
    "    yellow_region = v_crop[12:18, 2:12]\n",
    "    green_region = v_crop[22:28, 2:12]\n",
    "    \n",
    "    total_area = v_crop.shape[0] * v_crop.shape[1]\n",
    "    \n",
    "    sum_red = np.sum(red_region)\n",
    "    sum_yellow = np.sum(yellow_region)\n",
    "    sum_green = np.sum(green_region)\n",
    "    \n",
    "    # Each region normalized: dividing sum of each color region pixel values by total number of pixels\n",
    "    norm_red = sum_red / total_area\n",
    "    norm_yellow = sum_yellow / total_area\n",
    "    norm_green = sum_green / total_area\n",
    "    \n",
    "    \n",
    "    return [norm_red, norm_yellow, norm_green]\n",
    "\n",
    "\n",
    "# Function to calculate average rgb values of the image\n",
    "def avg_rgb(rgb_image):\n",
    "    \n",
    "    sum_red = np.sum(rgb_image[:,:,0])\n",
    "    sum_green = np.sum(rgb_image[:,:,1])\n",
    "    sum_blue = np.sum(rgb_image[:,:,2])\n",
    "    \n",
    "    total = rgb_image.shape[0] * rgb_image.shape[1]\n",
    "    \n",
    "    avg_red = sum_red / total\n",
    "    avg_green = sum_green / total\n",
    "    avg_blue = sum_blue / total\n",
    "    \n",
    "    return [avg_red, avg_green, avg_blue]\n",
    "\n",
    "\n",
    "\n",
    "def create_feature(rgb_image):\n",
    "    \n",
    "    brightness = norm_region_brightness(rgb_image)\n",
    "    \n",
    "    feature = brightness\n",
    "    \n",
    "    return feature\n",
    "\n",
    "# create_feature(test_im)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## (Optional) Create more features to help accurately label the traffic light images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "# (Optional) Add more image analysis and create more features\n",
    "def create_feature_2(rgb_image):\n",
    "    \n",
    "#     plt.imshow(rgb_image)\n",
    "\n",
    "    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)\n",
    "    \n",
    "    lower_range = np.array([10, 10, 5])\n",
    "    upper_range = np.array([255, 80, 190])\n",
    "    \n",
    "    mask = cv2.inRange(hsv, lower_range, upper_range)\n",
    "#     plt.imshow(mask)\n",
    "    \n",
    "    masked_image = np.copy(hsv)\n",
    "    masked_image[mask != 0] = [0, 0, 0]\n",
    "#     plt.imshow(masked_image)\n",
    "    \n",
    "    image_rgb = np.copy(masked_image)\n",
    "    rgb = cv2.cvtColor(image_rgb, cv2.COLOR_HSV2RGB)\n",
    "#     plt.imshow(rgb)\n",
    "    \n",
    "    image_crop = np.copy(rgb)\n",
    "    top_bottom = 1\n",
    "    left_right = 9\n",
    "    image_crop = rgb[top_bottom:-top_bottom, left_right:-left_right, :]\n",
    "    plt.imshow(image_crop)\n",
    "    \n",
    "    \n",
    "    image_avg_rgb = avg_rgb(image_crop)\n",
    "    \n",
    "    feature = image_avg_rgb\n",
    "   \n",
    "    return feature\n",
    "\n",
    "# create_feature_2(test_im)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## (QUESTION 1): How do the features you made help you distinguish between the 3 classes of traffic light images?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "Feature 1:\n",
    "V channel image is divided into 3 regions (red/yellow/green) as per the row and column count. Each region's brightness is calculated and compared. The most bright region tells us what color the signal is. Since the positions of red, yellow and green lights are fixed, this technique is very reliable.\n",
    "\n",
    "Feature 2:\n",
    "I converted the rgb image to hsv color space after which I created a mask for the resulting image. Masking is better done in hsv color space given the respective image database. Masked image is then converted into rgb color space and cropped.\n",
    "Total red, green and blue pixels are calculated in a given image and the label is estimated accordingly whether the red number is high or green (by a significant margin). Yellow is decided where the number of red and green pixels are close to each other (within a certain range)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# 4. Classification and Visualizing Error\n",
    "\n",
    "Using all of your features, write a function that takes in an RGB image and, using your extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!\n",
    "\n",
    "You are encouraged to write any helper functions or visualization code that you may need, but for testing the accuracy, make sure that this `estimate_label` function returns a one-hot encoded label."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='task8'></a>\n",
    "### (IMPLEMENTATION): Build a complete classifier "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function should take in RGB image input\n",
    "# Analyze that image using your feature creation code and output a one-hot encoded label\n",
    "\n",
    "\n",
    "##### This classifier uses V channel brightness to identify the 3 color signals ########\n",
    "\n",
    "\n",
    "def estimate_label(rgb_image):\n",
    "    \n",
    "    ## TODO: Extract feature(s) from the RGB image and use those features to\n",
    "    ## classify the image and output a one-hot encoded label\n",
    "    \n",
    "    feature = create_feature(rgb_image)\n",
    "    \n",
    "    predicted_label = [0, 0, 0]\n",
    "    \n",
    "    if np.argmax(feature) == 0:\n",
    "        # red\n",
    "        predicted_label[0] = 1\n",
    "        \n",
    "    elif np.argmax(feature) == 1:\n",
    "        # yellow\n",
    "        predicted_label[1] = 1\n",
    "        \n",
    "    else:\n",
    "        # green\n",
    "        predicted_label[2] = 1\n",
    "    \n",
    "    return predicted_label "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function should take in RGB image input\n",
    "# Analyze that image using your feature creation code and output a one-hot encoded label\n",
    "\n",
    "\n",
    "##### This classifier uses RGB colors to identify the 3 color signals ########\n",
    "\n",
    "\n",
    "def estimate_label_2(rgb_image):\n",
    "    \n",
    "    ## TODO: Extract feature(s) from the RGB image and use those features to\n",
    "    ## classify the image and output a one-hot encoded label\n",
    "    \n",
    "    feature = create_feature_2(rgb_image)\n",
    "    \n",
    "    red_num = feature[0]\n",
    "    green_num = feature[1]\n",
    "    blue_num = feature[2]\n",
    "    \n",
    "    \n",
    "    # predicted_label = [red, yellow, green]\n",
    "    predicted_label = [0, 0, 0]\n",
    "    \n",
    "\n",
    "    if abs(red_num - green_num) < 0.5:\n",
    "        # yellow\n",
    "        predicted_label[1] = 1\n",
    "    \n",
    "    elif red_num > green_num:\n",
    "        # red\n",
    "        predicted_label[0] = 1\n",
    "        \n",
    "    # for those very blurry images where blue color pixels were calculated to be higher than green ones\n",
    "    # and it was interfering with true color selection\n",
    "    elif blue_num > 215 and red_num < green_num:\n",
    "        # red\n",
    "        predicted_label[0] = 1\n",
    "        \n",
    "    else:\n",
    "        # green\n",
    "        predicted_label[2] = 1\n",
    "    \n",
    "    return predicted_label "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing the classifier\n",
    "\n",
    "Here is where we test your classification algorithm using our test set of data that we set aside at the beginning of the notebook! This project will be complete once you've pogrammed a \"good\" classifier.\n",
    "\n",
    "A \"good\" classifier in this case should meet the following criteria (and once it does, feel free to submit your project):\n",
    "1. Get above 90% classification accuracy.\n",
    "2. Never classify a red light as a green light. \n",
    "\n",
    "### Test dataset\n",
    "\n",
    "Below, we load in the test dataset, standardize it using the `standardize` function you defined above, and then **shuffle** it; this ensures that order will not play a role in testing accuracy.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using the load_dataset function in helpers.py\n",
    "# Load test data\n",
    "TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)\n",
    "\n",
    "# Standardize the test data\n",
    "STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)\n",
    "\n",
    "# Shuffle the standardized test data\n",
    "random.shuffle(STANDARDIZED_TEST_LIST)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Determine the Accuracy\n",
    "\n",
    "Compare the output of your classification algorithm (a.k.a. your \"model\") with the true labels and determine the accuracy.\n",
    "\n",
    "This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 99.66329966329967 %\n",
      "Number of misclassified images = 1 out of 297\n"
     ]
    }
   ],
   "source": [
    "# Constructs a list of misclassified images given a list of test images and their labels\n",
    "# This will throw an AssertionError if labels are not standardized (one-hot encoded)\n",
    "\n",
    "def get_misclassified_images(test_images):\n",
    "    # Track misclassified images by placing them into a list\n",
    "    misclassified_images_labels = []\n",
    "\n",
    "    # Iterate through all the test images\n",
    "    # Classify each image and compare to the true label\n",
    "    for image in test_images:\n",
    "\n",
    "        # Get true data\n",
    "        im = image[0]\n",
    "        true_label = image[1]\n",
    "        assert(len(true_label) == 3), \"The true_label is not the expected length (3).\"\n",
    "\n",
    "        # Get predicted label from your classifier\n",
    "        predicted_label = estimate_label(im)\n",
    "        assert(len(predicted_label) == 3), \"The predicted_label is not the expected length (3).\"\n",
    "\n",
    "        # Compare true and predicted labels \n",
    "        if(predicted_label != true_label):\n",
    "            # If these labels are not equal, the image has been misclassified\n",
    "            misclassified_images_labels.append((im, predicted_label, true_label))\n",
    "            \n",
    "    # Return the list of misclassified [image, predicted_label, true_label] values\n",
    "    return misclassified_images_labels\n",
    "\n",
    "\n",
    "# Find all misclassified images in a given test set\n",
    "MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)\n",
    "\n",
    "# Accuracy calculations\n",
    "total = len(STANDARDIZED_TEST_LIST)\n",
    "num_correct = total - len(MISCLASSIFIED)\n",
    "accuracy = num_correct/total\n",
    "\n",
    "print('Accuracy: ' + str(accuracy*100) + \" %\")\n",
    "print(\"Number of misclassified images = \" + str(len(MISCLASSIFIED)) +' out of '+ str(total))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='task9'></a>\n",
    "### Visualize the misclassified images\n",
    "\n",
    "Visualize some of the images you classified wrong (in the `MISCLASSIFIED` list) and note any qualities that make them difficult to classify. This will help you identify any weaknesses in your classification algorithm."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Label [red, yellow, green]:  [1, 0, 0]\n",
      "True Label [red, yellow, green]:  [0, 0, 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f29b1a3a940>"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f29b1def7f0>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize misclassified example(s)\n",
    "## TODO: Display an image in the `MISCLASSIFIED` list \n",
    "## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as\n",
    "\n",
    "image_num = 0\n",
    "mis_image = MISCLASSIFIED[image_num][0]\n",
    "mis_predicted_label = MISCLASSIFIED[image_num][1]\n",
    "mis_true_label = MISCLASSIFIED[image_num][2]\n",
    "\n",
    "print(\"Predicted Label [red, yellow, green]: \", mis_predicted_label)\n",
    "print(\"True Label [red, yellow, green]: \", mis_true_label)\n",
    "\n",
    "plt.imshow(mis_image)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "<a id='question2'></a>\n",
    "## (Question 2): After visualizing these misclassifications, what weaknesses do you think your classification algorithm has? Please note at least two."
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "**Answer:**\n",
    "Weakness 1: Daylight glare at the top of the image makes the red region brightness count more than what it is supposed to be.\n",
    "Weakness 2: Shape of the green signal (being an arrow and not a circle) reduces the green region brightness count since number of the bright pixels is less in that region."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test if you classify any red lights as green\n",
    "\n",
    "**To pass this project, you must not classify any red lights as green!** Classifying red lights as green would cause a car to drive through a red traffic light, so this red-as-green error is very dangerous in the real world. \n",
    "\n",
    "The code below lets you test to see if you've misclassified any red lights as green in the test set. **This test assumes that `MISCLASSIFIED` is a list of tuples with the order: [misclassified_image, predicted_label, true_label].**\n",
    "\n",
    "Note: this is not an all encompassing test, but its a good indicator that, if you pass, you are on the right track! This iterates through your list of misclassified examples and checks to see if any red traffic lights have been mistakenly labelled [0, 1, 0] (green)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "**<span style=\"color: green;\">TEST PASSED</span>**"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Importing the tests\n",
    "import test_functions\n",
    "tests = test_functions.Tests()\n",
    "\n",
    "if(len(MISCLASSIFIED) > 0):\n",
    "    # Test code for one_hot_encode function\n",
    "    tests.test_red_as_green(MISCLASSIFIED)\n",
    "else:\n",
    "    print(\"MISCLASSIFIED may not have been populated with images.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Improve your algorithm!\n",
    "\n",
    "**Submit your project after you have completed all implementations, answered all questions, AND when you've met the two criteria:**\n",
    "1. Greater than 90% accuracy classification\n",
    "2. No red lights classified as green\n",
    "\n",
    "If you did not meet these requirements (which is common on the first attempt!), revisit your algorithm and tweak it to improve light recognition -- this could mean changing the brightness feature, performing some background subtraction, or adding another feature!\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### Going Further (Optional Challenges)\n",
    "\n",
    "If you found this challenge easy, I suggest you go above and beyond! Here are a couple **optional** (meaning you do not need to implement these to submit and pass the project) suggestions:\n",
    "* (Optional) Aim for >95% classification accuracy.\n",
    "* (Optional) Some lights are in the shape of arrows; further classify the lights as round or arrow-shaped.\n",
    "* (Optional) Add another feature and aim for as close to 100% accuracy as you can get!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

