#!/usr/bin/env python
# coding: utf-8

# In[115]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[116]:


IMAGE_SIZE=256
BATCH_SIZE=32


# In[117]:


dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    
)


# In[118]:


class_names=dataset.class_names
class_names


# In[119]:


len(dataset)


# In[120]:


plt.figure(figsize=(12,5))
for image_batch,label_batch in dataset.take(1):
    for i in range(10):
        ax=plt.subplot(2,5,i+1)
        plt.title(class_names[label_batch[i]])
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")


# In[121]:


train_ds=dataset.take(54)


# In[122]:


len(train_ds)


# In[123]:


val_and_test=dataset.skip(54)


# In[124]:


len(val_and_test)


# In[125]:


val_ds=val_and_test.take(7)
test_ds=val_and_test.skip(7)


# In[126]:


len(val_ds),len(test_ds)


# In[127]:


# train_ds
# val_ds
# test_ds


# In[128]:


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[129]:


resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),  
    tf.keras.layers.Rescaling(1.0/255)  
])


# In[130]:


data_augmentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])


# In[ ]:





# In[ ]:





# In[139]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    resize_and_rescale,
    data_augmentation,  

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  
])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[140]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[156]:


history = model.fit(train_ds, epochs=10, batch_size=BATCH_SIZE, verbose=1, validation_data=val_ds)


# In[157]:


import numpy as np
for images_batch,label_batch in test_ds.take(1):
    first_image=images_batch[0].numpy().astype('uint8')
    first_label=label_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label : ", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label : ",class_names[np.argmax(batch_prediction[0])])


# In[158]:


def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence= round(100 * (np.max(predictions[0])),2)
    return predicted_class, confidence


# In[159]:


plt.figure(figsize=(15,20))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))

        predicted_class,confidence=predict(model,images[i].numpy())
        actual_class=class_names[labels[i]]
        plt.title(f"Actual label : {actual_class} ,\n Predicted : {predicted_class}, \n Confidence : {confidence} \n {actual_class==predicted_class}")
        plt.axis("off")


# In[ ]:




