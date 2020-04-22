# Title
> summary


## [fastai2 Series] ```Transform``` and ```ItemTransform``` Illustrated with Tiny COCO Dataset
fastai2 has been released for a while! For those readers who are new to fastai, it is a deep learning framework that wrap on top of PyTorch. The framework offers a bunch of out-of-box functionality, such as callbacks and learning rate finder, which helps deep learning users accelerate their experiments' cycle and make their codebase more readable. The framework and its community is under rapidly growth. fastai2 is one of their latest release. 

Compared to version 1, fastai2 has undergone a rapid changes in low-level and mid-level API. Many new concepts have been introduced in fastai2. For example, classes such as ```Transform```, ```Pipeline```, ```TfmdLists```, ```Datasets``` and ```DataLoaders``` are introduced to handle data processing pipeline.

While it may be intimidating to learn all of these concepts, let's try to break them down and learn them one by one. As a first post talking about fastai2, I will introduce ```Transform``` and ```ItemTransform``` with an illustrating example.

### 0. Prerequisites

To work out the code in this post, please install ```fastai2``` [here](https://github.com/fastai/fastai2). You are recommended to install ```fastai2``` in Linux. 

```python
import json

from PIL import Image
import matplotlib.patches as patches

import torch
import torchvision
from torch.utils.data import Dataset


import fastai2
from fastai2.vision.all import *
```

```python
print(f'fastai2 version: {fastai2.__version__}')
print(f'pytorch version: {torch.__version__}')
```

    fastai2 version: 0.0.17
    pytorch version: 1.4.0


### 1. Prepare Our Dataset
In this post, we use tiny COCO dataset from fastai2 as an example. You can easily download the dataset with the code snippet belows. 

The data directory has the following structure:  

```
coco_tiny
│   train.json    
│
└─ train
         000000291816.jpg
         000000285170.jpg
         000000295850.jpg
         ...
```

```python
src = untar_data(URLs.COCO_TINY)
src.ls()
```




    (#2) [Path('/userhome/34/h3509807/.fastai/data/coco_tiny/train'),Path('/userhome/34/h3509807/.fastai/data/coco_tiny/train.json')]



```python
(src/'train').ls(3)
```




    (#3) [Path('/userhome/34/h3509807/.fastai/data/coco_tiny/train/000000291816.jpg'),Path('/userhome/34/h3509807/.fastai/data/coco_tiny/train/000000285170.jpg'),Path('/userhome/34/h3509807/.fastai/data/coco_tiny/train/000000295850.jpg')]



### 2. Understand Our Dataset
COCO_TINY is a dataset for object detection problem. In object detection problem, we have to localize a set of objects in an image with bounding boxes and identify their classes. 

```train.json``` record the following information:
1. Class labels (in key ```'categories'```)
2. Mapping from image id to image filename (in key ```'images'```)
3. Information of all bounding boxes (in key ```'annotations'```).

In particular, information of all bounding boxes are recorded in the value of ```'annotations'```. The value of ```'annotations'``` is a list of dictionary, with each dictionary representing one bounding box. In each dictionary, the ```'image_id'``` tells you which image this bounding box belongs to. The key ```'bbox'``` contains a list of 4 entries (i.e. ```[x, y, w, h]```). They are respectively the top left x position, top left y position, width and height of the bounding box. Finally, the key ```'categoy_id'``` tells you the class of the object enclosed by the bounding box.

```python
meta = json.load(open(src/'train.json'))
meta.keys()
```




    dict_keys(['categories', 'images', 'annotations'])



```python
meta['categories']
```




    [{'id': 62, 'name': 'chair'},
     {'id': 63, 'name': 'couch'},
     {'id': 72, 'name': 'tv'},
     {'id': 75, 'name': 'remote'},
     {'id': 84, 'name': 'book'},
     {'id': 86, 'name': 'vase'}]



```python
meta['images'][:3]
```




    [{'id': 542959, 'file_name': '000000542959.jpg'},
     {'id': 129739, 'file_name': '000000129739.jpg'},
     {'id': 153607, 'file_name': '000000153607.jpg'}]



```python
meta['annotations'][:3]
```




    [{'image_id': 542959, 'bbox': [32.52, 86.34, 8.53, 9.41], 'category_id': 62},
     {'image_id': 542959, 'bbox': [98.12, 110.52, 1.95, 4.07], 'category_id': 86},
     {'image_id': 542959, 'bbox': [91.28, 51.62, 3.95, 5.72], 'category_id': 86}]



```python
def get_img_ids(json_path):
    meta = json.load(open(json_path, 'r'))
    return L(meta['images']).map(lambda d: d['id'])
```

```python
img_ids = get_img_ids(src / 'train.json')
img_ids
```




    (#200) [542959,129739,153607,329258,452866,162156,53058,400976,31971,67966...]



### 3. Introduce ```Transform```
```Transform``` is one of the base classes in fastai2 framework for data processing. It is used to define each atomic transformation that we apply on our dataset. We can have many ```Transform``` objects responsible for different data processing steps and we compose them by ```Pipeline``` (will be covered later). It is pretty similar to ```torchvision.transforms``` module, except that it has more extended functionalities. 

While fastai2 library has provided a lot of prebuilt generaic subclasses of ```Transform``` for users to use out-of-the-box. (e.g. ```ToTensor```, ```Resize```), in some scenarios the transformation that you want to apply on the dataset may not be something generic that you could find in the prebuilt. In this case you can write your own subclass of ```Transform```. fastai2 has provided you a high flexibility to customize your data transformation step. As an example, let's try to write a simple subclass of ```Transform``` for our tiny COCO dataset!

### 4a. Let's Write a ```Transform``` Subclass for Our Dataset!

We will write a subclass of ```Transform``` to apply a simple transform on our tiny COCO dataset: given an image id, we would like to read the corresponding image and its bounding boxes. For the bounding boxes, we would like to know its location in the image and the category of the object bounded by the boxes. 

```get_mappings``` is a simple helper function for reading the meta-data of our dataset. The meta-data are essentially a set of mappings that facilitate us to access the image and its bounding boxes according to image id. Besides that, we also have a mapper for category. 

```python
from collections import defaultdict

def get_mappings(src):
    """ idx: class index, id: image id """
    json_path = src / 'train.json'
    meta = json.load(open(json_path, 'r'))
    
    idx2name = dict()
    for d in meta['categories']:
        idx2name[d['id']] = d['name']
        
    id2fn = dict()
    for d in meta['images']:
        id2fn[d['id']] = d['file_name']
    
    id2bboxs = defaultdict(list)
    for bbox in meta['annotations']:
        id2bboxs[bbox['image_id']].append(bbox)
    return idx2name, id2fn, id2bboxs
```

We name our subclass as ```COCOTransform``` and it inherents from base class ```Transform```. Whenever we write a subclass of ```Transform```, there are a few methods we should write:

**```setups``` method**  
    One of the methods you should write is ```setups``` method. It is optional to write but good to have because it could help book keep the state of the transformation. In this example, we initialize different mappings in ```setups``` and store them as attributes of ```COCOTransform```. The mappings are useful for the transformation that we do in ```encodes```. Note that ```setups``` has to be called after we have initialized a ```COCOTransform``` object. After that we could apply the object on our dataset.


**```encodes``` method**  
Another method you should write is ```encodes```. It is necessary to define ```encodes``` method for your subclass. We would write the transformation steps that we would like to apply on our dataset here. In this example, we simply read the image as a ```PILImage``` and get its bounding boxes as a list. Note that the list of bounding boxes are augmented with its category represented by an index (will be discussed later). The method returns a tuple of image and its bounding boxes. Note that ```tuple``` as a return is important here because ```Transform``` has some magic tricks when working with ```tuple``` (you will see that magic soon!). One caveat is that ```encodes``` is called whenever you call a ```COCOTransform``` object on your data (i.e. ```tfm(img_ids[0])```). The concept is similar to the ```forward``` method in ```torch.nn.Module```.


As a remark about category mapping, you will notice we have several category mappings (i.e. ```self.vocab```, ```self.o2i```, ```self.idx2obj```) define in ```COCOTransform```. ```train.json``` provides a category mapper from a category integer (e.g. 62, 63, ..., 86) to a category name (e.g. chair, couch, ...,vase). The mapper is stored in ```self.idx2obj```. Such mapping is useful especially when we want to sample check the exact category name of each bounding box because they are originally recorded as a category integer. 

 On top of that, we also prepare a mapping ```self.o2i``` that tries to index each category integer (e.g. 0, 1, ..., 5). It maps each category integer to a category index. Such 0-based index would be more convenient to work with in modelling stage because the index will be expanded as a one-hot vector before feeding into model. On the other hand, ```self.o2i``` is used to convert an index back to a category integer. 

```python
# bbox [x, y, w, h] need to be normalize

class COCOTransform(Transform):
    def setups(self, src):
        self.src = src
        self.idx2obj, self.id2fn, self.id2bboxs = get_mappings(src)
        vals = list(self.idx2obj.keys())
        self.vocab, self.o2i = uniqueify(vals, sort = True, bidir = True)
    
    def encodes(self, img_id):
        fn = self.id2fn[img_id]
        img_path = src/'train'/fn
        bbox_data = self.id2bboxs[img_id]
        bboxs = [d['bbox'] + [self.o2i[d['category_id']]] for d in bbox_data]
        return (PILImage.create(img_path), bboxs)
```

```python
tfm = COCOTransform()
tfm.setups(src)
img, bboxs = tfm(img_ids[0])
type(img), img.shape, type(bboxs), len(bboxs)
```




    (fastai2.vision.core.PILImage, (128, 128), list, 6)



```python
bboxs
```




    [[32.52, 86.34, 8.53, 9.41, 0],
     [98.12, 110.52, 1.95, 4.07, 5],
     [91.28, 51.62, 3.95, 5.72, 5],
     [110.48, 110.82, 14.55, 15.22, 0],
     [96.63, 50.18, 18.67, 13.46, 0],
     [0.69, 111.73, 11.8, 13.06, 0]]



```python
img
```




![png](/images/2020-04-23-fastai-series1/output_21_0.png)



### 4b. Add a Reversible Transform to our Subclass

One benefit about using subclass of ```Transform``` is that it makes our data processing pipeline more modular and hence easier to manage. Usually in order to train a deep learning, we have to do a lot of data processing steps. Below are some typical steps:
- read an image file (e.g. transform from a file path to a PIL.Image object)
- convert our image to a tensor (e.g. transform from PIL.Image object to torch.Tensor)
- normalize our data (e.g. normalize torch.Tensor object)
- converting our class into an one-hot vector 

In addition, for the sake of sanity check, we usually want to do inspection on the intermediate output from each transformation step. But sadly, some intermediate outputs are just unfriendly for inspection or visualization. (e.g. it's hard to visualize an image of a form as ```torch.Tensor```) As a result, we would also write some helper functions to convert those intermediate outputs into a more readable form. Taking all these into accounts, you could imagine we have to write a lot of helper functions to do all sort of data processing work. If not properly managed, the codebase of our data pipeline will quickly turn into a mess.

```Transform``` could help with the issue by allowing you to define a method that does a reverse transform of ```encodes```. Here we introduce one more method that you should write in your subclass of ```Transform```.

**```decodes``` method**  
We could write our reverse transform in ```decodes``` method. While ```encodes``` method converts data to a form that is easy for a model to process, ```decodes``` method serves to convert data to a form is easy for us to inspect or visualize. In ```decodes``` method, we could return anything that is easy for us to interpret. We usually apply ```decodes``` on the output from ```encodes```. That's why we call it reverse transform. Let's add ```encodes``` method to our ```COCOTransform```. In the example, we take the ```tuple``` output from ```encodes``` as the input of ```decodes```. We notice the bounding boxes data returned by ```encodes``` is hard to read because it is expressed as a category index. We convert it back to a category name in our ```decodes``` method. One thing to note is that while we define reverse transformation in ```decodes```, we actually apply ```decode``` on our dataset. You are not suggested to manually call ```decodes``` from a ```COCOTransform``` object. It is not supposed to be directly called by user.

**```ItemTransform``` v.s. ```Transform```**  
Besides adding ```decodes``` method to ```COCOTransform```, note that we also change the base class from ```Transform``` to ```ItemTransform```. They essentially provide the same functionality except they are different in the way they read the input in ```decodes``` and ```encodes```. If we subclass in ```Transform```, whenever we feed a ```tuple``` to ```encodes``` or ```decodes``` method, it will take each individual entry as an input. If we subclass in ```ItemTransform```, it will read the whole ```tuple``` as an input. In this example, we want ```decodes``` to take the whole ```tuple``` as an input so we use ```ItemTransform```.

```python
class COCOTransform(ItemTransform):
    def setups(self, src):
        self.src = src
        self.idx2obj, self.id2fn, self.id2bboxs = get_mappings(src)
        vals = list(self.idx2obj.keys())
        self.vocab, self.o2i = uniqueify(vals, sort = True, bidir = True)
    
    def encodes(self, img_id):
        fn = self.id2fn[img_id]
        img_path = src/'train'/fn
        bbox_data = self.id2bboxs[img_id]
        bboxs = [d['bbox'] + [self.o2i[d['category_id']]] for d in bbox_data]
        return (PILImage.create(img_path), bboxs)
    
    def decodes(self, x):
        img, bboxs = x
        bboxs = [[x0, y0, w, h, self.idx2obj[self.vocab[c]]] for x0, y0, w, h, c in bboxs]
        return (img, bboxs)
```

```python
tfm = COCOTransform()
tfm.setups(src)
x_enc = tfm(img_ids[0])
x_dec = tfm.decode(x_enc)
img, bboxs = x_dec
type(img), type(bboxs), len(bboxs)
```




    (fastai2.vision.core.PILImage, list, 6)



As shown below, we managed to convert category indexes of our bounding boxes to category names by calling ```decode``` on our data. Thus far you can see the code clarity brought by ```Transform``` -- By grouping a forward transform and its corresponding reverse transform in the same class, it is more handy to transform our data back and forth, and the code is more readable.

```python
bboxs
```




    [[32.52, 86.34, 8.53, 9.41, 'chair'],
     [98.12, 110.52, 1.95, 4.07, 'vase'],
     [91.28, 51.62, 3.95, 5.72, 'vase'],
     [110.48, 110.82, 14.55, 15.22, 'chair'],
     [96.63, 50.18, 18.67, 13.46, 'chair'],
     [0.69, 111.73, 11.8, 13.06, 'chair']]



```python
x_dec[0]
```




![png](/images/2020-04-23-fastai-series1/output_27_0.png)



### 4c. Type Dispatching in Transform

One thing powerful about ```Transform``` (and ```ItemTransform```) is that it supports type dispatching. Type dispatching means the same function could have different behavior according to the type of the input. Such design feature occurs in other programming langugages, such as Julia, but it is not introduced in Python. By some means, ```Transform``` does get a hack to make such behavior work in Python. It is one of the major changes in fastai2 that kind of violate our usual understanding of Python behaviors, but at the same time enable us to make many tricks in data processing. We could magically inherit such type dispatching feaature by subclassing ```Transform```.

Remember I mentioned in last session that ```Transform``` reads each individual entry of a ```tuple``` as an input of ```encodes``` and ```decodes```? When such feature is combined with type dispatching, we could do a lot of amazing tricks! Let's rewrite ```COCOTransform``` again, but this time we use ```Transform``` as base class and we apply type dispatching in ```decodes``` method.

```python
class COCOTransform(Transform):
    def setups(self, src):
        self.src = src
        self.idx2obj, self.id2fn, self.id2bboxs = get_mappings(src)
        vals = list(self.idx2obj.keys())
        self.vocab, self.o2i = uniqueify(vals, sort = True, bidir = True)
    
    def encodes(self, img_id):
        fn = self.id2fn[img_id]
        img_path = src/'train'/fn
        bbox_data = self.id2bboxs[img_id]
        bboxs = [d['bbox'] + [self.o2i[d['category_id']]] for d in bbox_data]
        return (PILImage.create(img_path), bboxs)
    
    def decodes(self, x: PILImage):
        return x
    
    def decodes(self, x: list):
        x = [[x0, y0, w, h, self.idx2obj[self.vocab[c]]] for x0, y0, w, h, c in x]
        return x
```

```python
tfm = COCOTransform()
tfm.setups(src)
x_enc = tfm(img_ids[0])
x_dec = tfm.decode(x_enc)
img, bboxs = x_dec
type(img), type(bboxs), len(bboxs)
```




    (fastai2.vision.core.PILImage, list, 6)



```python
bboxs
```




    [[32.52, 86.34, 8.53, 9.41, 'chair'],
     [98.12, 110.52, 1.95, 4.07, 'vase'],
     [91.28, 51.62, 3.95, 5.72, 'vase'],
     [110.48, 110.82, 14.55, 15.22, 'chair'],
     [96.63, 50.18, 18.67, 13.46, 'chair'],
     [0.69, 111.73, 11.8, 13.06, 'chair']]



```python
x_dec[0]
```




![png](/images/2020-04-23-fastai-series1/output_32_0.png)



We get the same result as the last session!

### 5. Conclusion
In this post, we have learnt how to write a subclass of ```Transform``` (and ```ItemTransform```) to process tiny COCO dataset. The data processing steps we have done in this post are far from the end, there are more steps we need to do on input data and targets before they are ready for feeding into an object detection model (e.g. permute tensor axis for input data, normalize input data, turning class index into a one-hot tensor in targets, grouping data as a batch ... etc.), but I hope at this point the post is enough to demonstrate how we could use ```Transform``` to process our dataset!

### 6. Reference
1. [Practical-Deep-Learning-for-Coders-2.0/Computer Vision/06_Object_Detection.ipynb](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer%20Vision/06_Object_Detection.ipynb)
2. [fastai2/nbs/10_tutorial.pets.ipynb](https://github.com/fastai/fastai2/blob/master/nbs/10_tutorial.pets.ipynb)
3. [Part 1. Preparing data before training YOLO v2 and v3.](https://blog.goodaudience.com/part-1-preparing-data-before-training-yolo-v2-and-v3-deepfashion-dataset-3122cd7dd884)
