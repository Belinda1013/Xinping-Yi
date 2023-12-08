#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks
# 

# In[ ]:


https://github.com/Belinda1013/Xinping-Yi


# # Getting the Data

# In[54]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL url of where you are getting your data from
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url) #Function downloads the Shakespeare text from the specified URL and saves it to a local file named shakespeare.txt
with open(filepath) as f: #This line opens the downloaded file using the open function and assigns it the variable “f“
    shakespeare_text = f.read() #This line read the file text and assigns it the variable "shakespeare_text"


# In[57]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# In[ ]:


#This function prints the first 80 characters of Shakespeare


# # Preparing the Data

# In[ ]:


#TextVectorization is a feature of TensorFlow that converts raw text into numeric vectors that can be used in machine learning models.
#It includes a series of preprocessing steps.


# In[ ]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",#Get character level encoding Instead of the default font-level encoding,
                                                   standardize="lower") #convert the text to lowercase sk)
text_vec_layer.adapt([shakespeare_text]) #The adapt method analyzes the provided text
encoded = text_vec_layer([shakespeare_text])[0]


# In[63]:


print(text_vec_layer([shakespeare_text]))


# In[74]:


#tensor is a multidimensional arrays These numbers are individually accessible via an index and can be indexed via multiple indices.
from IPython.display import Image #the inmage below shows what is tensor
Image ("t.jpg")
#shape=(1,1115394) shows the shape of tensor, and means it is a 2D tensor with one row and 1115394 columns
#dtype=int64 means the data type of the tensor elements is int64


# In[64]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# In[65]:


print(n_tokens, dataset_size)


# In[ ]:


#39 means the number of distinct characters in the text
#1115394 means the total number of characters in the dataset after drop tokens0 (pad) and 1 (unknown)


# In[ ]:


#the variable ds:Digital signal means that the independent variable is discrete


# In[76]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32): #Batch Size refers to the amount of data input to the model for training each time during the machine learning model training process.
    ds = tf.data.Dataset.from_tensor_slices(sequence) #here will creat tensorflow dataset from input
    ds = ds.window(length + 1, shift=1, drop_remainder=True) #here we create a dataset containing all windows of required length.
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed) #If the shuffle parameter is set to True,it shuffles the dataset using a buffer size of 100,000
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# In[80]:


length = 100 # here we set the window size is 100
tf.random.set_seed(42) #This sets the random seed to ensure dataset reproducibility.
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)


# # Building and Training the Model

# In[43]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16), # It means that each token in the vocabulary will be represented by a dense vector of 16 dimensions.
    tf.keras.layers.GRU(128, return_sequences=True), #this modle hase a GRU layer with 128 units
    tf.keras.layers.Dense(n_tokens, activation="softmax") #dense layer is a output layer and it has 39 units
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[90]:


from IPython.display import Image
Image ("s.jpg")


# In[82]:


#When a complete data set is passed through the neural network once and returned once, the process is called an epoch.
#As the number of epochs increases, the number of updates to the weights in the neural network also increases, and the curve changes from underfitting to overfitting.
from IPython.display import Image
Image ("ep.jpg")


# In[81]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# In[45]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[83]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42) #This will produce more diverse and interesting text.
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[ ]:


#To better control the diversity of generated text, we can divide logits by a number called temperature


# In[84]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[85]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[86]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[ ]:


# we can found when the temperature is more close to 0 will lead high probability characters


# In[87]:


print(extend_text("To be or not to be", temperature=0.01))


# In[88]:


print(extend_text("To be or not to be", temperature=1))


# In[89]:


print(extend_text("To be or not to be", temperature=100))

