#!/usr/bin/env python
# coding: utf-8

# #Week 1: "Getting started with Anaconda,Jupyter Notebook and Python".
#  
#  a)Because I think artificial intelligence technology is a very mainstream technology now, I am very interested in it. I wanted to know exactly how it worked so I joined the course.
#  
#  b)No, I don't have prior experince.
#  
#  c)First I want to know how artificial intelligence can draw beautiful paintings, and secondly I want to know whether artificial intelligence can create something new when it is fed enough knowledge. Finally, I want to know when an accident occurs because of artificial intelligence, who is blamed for the error.

# print("Hello,World")

# message = "I am good!"
# print (message)

# In[ ]:


https://github.com/Belinda1013/Xinping-Yi


# # Task 1-5 Use the Code Above to Answer the Following Questions

# message ="I am good!"+"I am hungry!"
# print (message)

# In[ ]:


#Here if I run the code above I will get "I am good! I am hungry! message is variable,
#if I define the message first then when I print, I don’t have to enter all the text again.


# message_1 ="I am good!"
# print (message_1*3)

# In[ ]:


#1*3 means print the message three times, so my output is I am good!I am good!I am good!


# message_2 ="I am good!"
# print (message_2 [0])

# message_3 ="I am good!"
# print (message_3 [2])

# In[ ]:


#the number here means statement prints the character at string index 0, which is the letter "I"
#If I change the number to 2 which is the letter "a"


# In[ ]:


#I think message is a good variable name,But if I use message all the time, it is easy to confuse, so I made some marks above.


# #**Task 1-6 A first look at importing library and packages**

# In[10]:


from IPython.display import *


# In[11]:


YouTubeVideo("https://www.youtube.com/watch?v=Px5nmExb-uw" )


# #**Week 2. "Exploring Data in Multiple Ways".**

# #**Task 3.1 "Using IPython.display to display images and audio"**

# In[2]:


from IPython.display import Image


# In[3]:


Image ("picture1.jpg")


# In[ ]:


#if I want to import the image I want from the python environment,first I need to write this function, 
#secondly I need to make sure I upload my image to my jupyter notebook folder, and then define the image variable
#After following the steps I successfully displayed the photo of the hedgehog


# In[14]:


from IPython.display import Audio


# In[ ]:


Audio ("Audio1")


# Audio("Audio2.ogg")    #The sound file audio2.ogg is owned by Artoffuge Mehmet Okonsar. 

# In[ ]:


#The principle of this part is the same as that of importing pictures, 
#and Audio1 cannot play smoothly here, but Audio2.ogg can play normally.
#I think it's because the file format of Audio1 is not suitable in this environment and therefore cannot run.


# #**Task 3.2 "Using the matplotlib library to look at picture as numerical data"**

# In[6]:


from matplotlib import pyplot
test_picture = pyplot.imread("index-1.png")
print("Numpy array of the image is:", test_picture)
pyplot.imshow(test_picture)


# In[ ]:


#This code load the image, print its numpy array representation, display the image, and then show the plot.
#Firstly


# #**Task 3.3 " Exploring scikit-learn (a.k.a sklearn)"**

# dir.load_iris
# dir.load_digits
# dir.load_breast_cancer

# In[ ]:


#Reason why chose these datasets
#First of all, because the iris data set is a relatively simple data set, it is easy for me to understand as a beginner.
#Secondly, the digital data set is more complicated than the iris data set. It has 64 functions and is more difficult to analyze than the first one.
#The last Wisconsin breast cancer dataset is different in kind than the previous two, it is a binary classification and is more complex than the previous two.
#Therefore, I chose these three relatively different data sets for research.


# iris_data = datasets.load_iris()
# iris_data.DESCR
# print(iris_data.DESCR)

# In[ ]:


#I got 4 features of iris
#The aim of this datesets is to predict the class of new samples based on these features. 


# #Task 3-4 Basic Data Exploration with Python library Pandas 

# In[19]:


from sklearn import datasets
import pandas

iris_data = datasets.load_iris()

iris_dataframe = pandas.DataFrame(data=iris_data['data'],columns = iris_data['feature_names'])


# In[21]:


iris_dataframe.head()


# In[ ]:


#import pandas menas create a pandas data frame from the data.
#The data parameter is a NumPy array containing the data, and the columns parameter is a list of strings containing feature names.
#Head command prints the first few rows of the DataFrame.


# #**Task 3-5: Thinking about data bias (home work)**

# In[ ]:


#First, I will analyze the distribution classification of the data set. If a data set is concentrated in certain areas or a specific group, then the data will be biased.
#Secondly, I will find my colleagues to help me analyze, because maybe I have some biases,it will be difficult for me to find my own biases, but others may observe them from the same perspective.
#I can also use tools to help me analyze the data, for example Google, IBM, and Microsoft have all released tools and guides to help analyze biases on many different data types. 
#In this way I can find out what data I missed so I can correct it.

