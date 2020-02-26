# Transfer Learning

Utilizing Keras built-in VGG-16 net module

Why there is a need for the include_top for the VGG16 ?
What - nesterov for the optimizer ?


A full list of pre-trained Keras models is available at: https://keras.io/applications/ 

```python
from keras.models import Model
from keras.preprocessing import image
from keras.optimizer import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input


#  pre- trained model and pre-trained weights on imagenet:

model=VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr= 0.1, decal =1e-6, momentum=0.9, nesterov=True)

# resize images to the VGG16 trained image format
im = cv2.resize(cv2.imread('steam-locomotive.jpg'),(224,224))
im = np.expand_dims(im ,axis=0)


# predict with the new model:
out = model.predict(im)
plt.plot(out.ravel())
plt.show()

print(np.argmax(out))
```



### Recycling pre-built deep learning models for extracting features


The key intuition is that, as the network learns to classify images into categories, each layer learns to identify the features that are necessary to do the final classification.

Lower layers identify lower order features such as color and edges, and higher layers compose these lower order features into higher order features such as shapes or objects.

Hence the intermediate layer has the capability to extract important features from an image, and these features are more likely to help in different kinds of classification. 

This has multiple advantages:   
- First, we can rely on publicly available large-scale training and transfer this learning to novel domains.  
- Second, we can save time for expensive large training.  
- Third, we can provide reasonable solutions even when we don't have a large number of training examples for our domain.  

We also get a good starting network shape for the task at hand, instead of guessing it.

###  Pre-trained model and pre-trained weights on imagenet:

```python
base_model =VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
	print( i , layer.name, layer.output_shape)

### new image:
img_path = 'cat.jpg'

img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x , axis=0)
x = preprocess_input(x)

### get features from  this block
features =  model.predict(x)
```

**Very deep inception-v3 net used for transfer learning**

Transfer learning is a very powerful deep learning technique which has more applications in different domains. The intuition is very simple and can be explained with an analogy. 

Suppose you want to learn a new language, say Spanish; then it could be useful to start from what you already know in a different language, say English.

Following this line of thinking, computer vision researchers now commonly use pre-trained CNNs to generate representations for novel tasks, where the dataset may not be large enough to train an entire CNN from scratch. 

Another common tactic is to take the pre-trained ImageNet network and then to fine-tune the entire network to the novel task.

**Inception-v3** net is a very deep ConvNet developed by Google. Keras implements the full network described in the following diagram and it comes pre-trained on ImageNet. The default input size for this model is 299 x 299 on three channels:

Suppose we have training dataset D in a domain, other than Imagenet. D has 1024 input features and 200  categories in output.


```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base_model from the pretrained
base_model=  InceptionV3(weights='imagenet', include_top=False)
```

**Include_top :** we do not include the top of the model because we want to finetune  ‘D’.  The top layer is dense with 1024 input and output is a softmax dense layer with 200 classes
x = GlobalAveragePooling2D()	
X is used to convert the input of the new images to the correct shape of the dense layer.
Last three layers:
	


When,   we are applying Include_top=False, that means we are removing the last three layers and exposing ‘mixed10’ layer so GlobalAveragePooling2D converts
(None, 8 , 8 , 2048)  > (None, 2048)
Where each value of the converted is the average value for each 8 x 8 subtensor.

```python
#  add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x) # lets add a fully connected first layer
x = Dense(1024, activation='relu') # and add a logistic layer  with 200 classes as the last layer

predictions = Dense(200, activation = 'softmax') (x) # models to train
model= Model(input = base_model.input,  output = predictions)

All convolutional models are pre-trained,, so we freeze them during the training of the model  

# that is freeze all the convolutional InceptionV3 layers
for layer in base_model.layers:
	layer.trainable=False
The model is then compiled and trained for a few epochs so that the top layers are trained:

# compile the model ( ***should be done, after setting the layers non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for few epochs model.fit_generator(...)

Then we freeze the top layers in inception and fine-tune some inception layer. In this example, we decide to freeze the first 172 layers (an hyperparameter to tune):
	
# we choose to train top-2 inception blocks, that is, we will freeze  the first # 172 layers and unfreeze the rest:

for layer in model.layers[:172]:
	layer.trainable =  False

for layer in model.layer[172:]:
	layers.trainable = True

The model is then recompiled for fine-tune optimization. We need to recompile the model for these modifications to take effect

# we use SGD with low learning rate:

from keras.optimizers import SGD
model.compile(optimizers=SGD(lr=0.0001,momentum=0.9,loss='categorical_crossentropy'))

# we train our model again and again (this time finetune the top 2 inception block)
# alongside the top Dense Layer
model.fit_generator(...)
```