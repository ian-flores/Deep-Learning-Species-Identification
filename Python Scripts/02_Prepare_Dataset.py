
# coding: utf-8

# # Prepare Dataset

# by Ian Flores & Alejandro Vega

# ### Loading Dependencies

# In[46]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from six.moves import cPickle as pickle
import os
import shutil
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display as disp
from IPython.display import Image as Im
from scipy import ndimage
import random
import sys

# In[47]:


## image size
size = 28
image_size = size

## Shifts
num_shifts = int(sys.argv[2])

## Number of imgs per class
min_imgs_per_class = 1

## Number of imgs per class after augmentation
min_augmentation = 20


# ### Cropping Spectrograms

# Given the architectures we are using in our models, we want all spectrograms to have the same size, because the models don't allow for dynamic size input.

# In[48]:


def squareAndGrayImage(image, size, path, species, name):
    # open our image and convert to grayscale
    # (needed since color channels add a third dimmension)
    im = Image.open(image).convert('L')
    # dimmensions of square image
    size = (size,size)
    # resize our image and adjust if image is not square. save our image
    squared_image = ImageOps.fit(im, size, Image.ANTIALIAS)
    squared_image.save(path + '/' + species + '/squared_' + name)
    #print(ndimage.imread(path + '/' + species + '/squared_' + name).shape)

def squareAndGrayProcess(size, dataset_path, new_dataset_path):
    # if our dataset doesn't exist create it, otherwise overwrite
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)
    else:
        shutil.rmtree(new_dataset_path)
        os.makedirs(new_dataset_path)

    # get a list of species folders in our dataset
    species_dataset = os.listdir(dataset_path)

    for species in species_dataset:
        os.makedirs(new_dataset_path + '/' + species)
        species_images = os.listdir(dataset_path + '/' + species)
        for image in species_images:
            image_path = dataset_path + '/' + species + '/' + image
            squareAndGrayImage(image_path, size, new_dataset_path, species, image)

dataset_path = '../dataset/spectrogram_roi_dataset'
new_dataset_path = '../dataset/squared_spectrogram_roi_dataset'
squareAndGrayProcess(size, dataset_path, new_dataset_path)


# In[49]:

if not os.path.exists("../dataset/augmented_spectrograms"):
    os.makedirs("../dataset/augmented_spectrograms")
else:
    shutil.rmtree("../dataset/augmented_spectrograms")
    os.makedirs("../dataset/augmented_spectrograms")
#os.mkdir("../dataset/augmented_spectrograms")


# In[23]:


from scipy.ndimage.interpolation import shift
## Have to find a way to create and copy the old directory #########

#To shift UP up to num_shifts pixels
for folder in os.listdir(new_dataset_path):
    species_pictures = os.listdir(new_dataset_path + '/' + folder)
    os.makedirs('../dataset/augmented_spectrograms' + '/' + folder)
    for image in species_pictures:
        the_image = np.asarray(Image.open(new_dataset_path + '/' + folder + '/' + image))
        for i in range(num_shifts):
            pre_image = the_image.reshape((size,size))
            shifted_image = shift(pre_image, [(i*(-1)), 0])
            shifted_image = Image.fromarray(shifted_image)
            shifted_image.save('../dataset/augmented_spectrograms/' + folder + '/' + 'shifted_up' + str(i) + '_' + image)
            shifted_image.close()


# In[24]:


#To shift down up to num_shifts pixels
for folder in os.listdir(new_dataset_path):
    species_pictures = os.listdir(new_dataset_path + '/' + folder)
    for image in species_pictures:
        the_image = np.asarray(Image.open(new_dataset_path + '/' + folder + '/' + image))
        for i in range(num_shifts):
            pre_image = the_image.reshape((size,size))
            shifted_image = shift(pre_image, [i, 0])
            shifted_image = Image.fromarray(shifted_image)
            shifted_image.save('../dataset/augmented_spectrograms/' + folder + '/' + 'shifted_down' + str(i) + '-' + image)
            shifted_image.close()


# In[25]:


#To shift to the left up to num_shifts pixels
for folder in os.listdir(new_dataset_path):
    species_pictures = os.listdir(new_dataset_path + '/' + folder)
    for image in species_pictures:
        the_image = np.asarray(Image.open(new_dataset_path + '/' + folder + '/' + image))
        for i in range(num_shifts):
            pre_image = the_image.reshape((size,size))
            shifted_image = shift(pre_image, [0, (i*(-1))])
            shifted_image = Image.fromarray(shifted_image)
            shifted_image.save('../dataset/augmented_spectrograms/' + folder + '/' + 'shifted_left' + str(i) + '-' + image)
            shifted_image.close()


# In[26]:


#To shift to the right up to num_shifts pixels
for folder in os.listdir(new_dataset_path):
    species_pictures = os.listdir(new_dataset_path + '/' + folder)
    for image in species_pictures:
        the_image = np.asarray(Image.open(new_dataset_path + '/' + folder + '/' + image))
        for i in range(num_shifts):
            pre_image = the_image.reshape((size,size))
            shifted_image = shift(pre_image, [0, i])
            shifted_image = Image.fromarray(shifted_image)
            shifted_image.save('../dataset/augmented_spectrograms/' + folder + '/' + 'shifted_right' + str(i) + '-' + image)
            shifted_image.close()


# In[27]:


new_dataset_path = '../dataset/augmented_spectrograms'


# In[28]:


# Function for displaying a random photo from each class in a dataset
def displaySamples(dataset_folders):
    # go through each class in the dataset
    dataset = os.listdir(dataset_folders)
    for folder in dataset:
        imgs_path = dataset_folders + '/' + folder
        imgs = os.listdir(imgs_path) # list all images in a class
        sample = dataset_folders + '/' + folder + '/' + imgs[np.random.randint(len(imgs))] # path for a random image from a dataset class
        name = sample.split('/')[-2]
        print(name, 'sample :')
        disp(Im(sample)) # display our sample
        print("========================================")

#print("Here's a random sample from each class in the training dataset:")
#displaySamples(new_dataset_path)



# In[29]:


def getDatasetFolders(dataset_path):
    folders = os.listdir(dataset_path)
    dataset_folders = []
    for folder in folders:
        dataset_folders.append(dataset_path + '/' + folder)
    return dataset_folders

dataset_folders = getDatasetFolders(new_dataset_path)


# In[30]:


pixel_depth = 255.0 # Number of levels per pixel.

def load_image(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    num_images = 0

    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            #print(image_data.shape)
            # our images are RGBA so we would expect shape MxNx4
            # see: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.imread.html
            if (image_data.shape != (image_size, image_size)):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    #if num_images < min_num_images:
     #   raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


# ### Pickling Data

# We want to pickle the data by species, allowing for control of the minimum images per class. Beware that this will drastically influence the performance of your model.

# In[33]:


def maybe_pickle(data_folders, min_num_images_per_class, pickles_path, force=False):

    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)
    else:
        shutil.rmtree(pickles_path)
        os.makedirs(pickles_path)

    dataset_names = []
    for folder in data_folders:
        class_name = folder.split('/')[-1] # species name
        set_filename = pickles_path + '/' + class_name + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            image_files = os.listdir(folder)
            count = 0
            for image in image_files:
                count +=1
            if count >= min_num_images_per_class:
                print('Pickling %s.' % set_filename)
                dataset = load_image(folder, min_num_images_per_class)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

    return dataset_names

pickles_path = '../dataset/pickle_data'
datasets = maybe_pickle(dataset_folders, min_imgs_per_class, pickles_path)


# In[34]:


pickles = getDatasetFolders('../dataset/pickle_data')
#print(datasets)
num_classes = len(pickles)
print(f'We have {num_classes} classes')


# ### Classes

# We have to evaluate the number of classes and how are they distributed. Also, observe which species has a higher frequency, etc.

# In[35]:


# Calculates the total of images per class
def class_is_balanced(pickles):
    total = 0
    for pckle in pickles:
        if (os.path.isfile(pckle)):
            pickle_class = pickle.load(open(pckle, "rb"))
        else:
            print("Error reading dataset %s. Exiting.", pickle_path)
            return -1
        class_name = pckle.split('/')[-1].split('.')[0]
        print("The total number of images in class %s is: %d" % (class_name, len(pickle_class)))
        total += len(pickle_class)
    print("For the dataset to be balanced, each class should have approximately %d images.\n" % (total / len(pickles)))
    return (total // len(pickles))

print("Let's see if the dataset is balanced:")
balance_num = class_is_balanced(pickles)


# In[36]:


def getBalancedClasses(pickle_files, balanced_num):
    pickle_paths = []
    total = 0
    for pckle in pickle_files:
        if (os.path.isfile(pckle)):
            pickle_class = pickle.load(open(pckle, "rb"))
        else:
            print("Error reading dataset %s. Exiting.", pickle_path)
            return -1
        if (len(pickle_class) >= balance_num):
            total += len(pickle_class)
            pickle_paths.append(pckle)
    return pickle_paths, total

true_pickles, total_balanced = getBalancedClasses(pickles, balance_num)

#print("balanced dataset is ", true_pickles)
print("Total Original Images are ", total_balanced)


# In[37]:


def getLargestClass(pickles):
    num_images = 0
    class_info = []
    for index,pckle in enumerate(pickles):
        if (os.path.isfile(pckle)):
            pickle_class = pickle.load(open(pckle, "rb"))
        else:
            print("Error reading dataset %s. Exiting.", pickle_path)
            return -1
        class_name = pckle.split('/')[-1].split('.')[0]
        if(len(pickle_class) > num_images):
            num_images = len(pickle_class)
            class_info = [index, class_name, num_images]

    print("Largest dataset is {} with {} images".format(class_info[1], class_info[2]))
    return class_info

#class_info = getLargestClass(true_pickles)


# In[38]:


def findMinClass(dataset_path):
    minm = float('inf')
    species = ''
    for folder in dataset_folders:
        images= os.listdir(folder)
        count = len(images)
        if (count < minm):
            minm = count
            species = folder.split('/')[-1]

    return (species, minm)


# In[40]:


# go through our pickles, load them, shuffle them, and choose class_size amount of the images
def makeSubClasses(class_size, pickle_path, pickle_files):

    # create path for folder of pickles of subsets of classes
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    else:
        shutil.rmtree(pickle_path)
        os.makedirs(pickle_path)

    # list of pickles of subsets of classes
    subclasses = []

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                name = pickle_path + pickle_file.split('/')[-1].split('.')[0] + '_subset.pickle'
                species_set = pickle.load(f) # set of images from species
                # let's shuffle the letters to have random subset
                np.random.shuffle(species_set)
                species_set = species_set[:class_size,:,:]
                try:
                    with open(name, 'wb') as f:
                        pickle.dump(species_set, f, pickle.HIGHEST_PROTOCOL)
                        subclasses.append(name)
                except Exception as e:
                    print('Unable to save data to', name, ':', e)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            pass
    return subclasses


# In[50]:


pickle_subclasses = makeSubClasses(min_augmentation, '../dataset/subclasess_pickle_data/', pickles)


# ### Training, Testing, and Validation Separation

# As with every implementation of Supervised Learning, we separate the dataset into three components. The training, the testing, and the validation dataset.

# In[51]:


# our dataset is now balanced. calculate our training, validation, and test dataset sizes
total_images = len(pickle_subclasses) * 56
print("We have a total of {}.".format(total_images))
print("We'll split them 70/15/15 for training, validation, and testing respectively.")
print("Training dataset size: {}".format(int(total_images*0.70)))
print("Validation dataset size: {}".format(int(total_images*0.15)))
print("Testing dataset size: {}".format(int(total_images*0.15)))


# In[52]:


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets_all(pickle_files, train_size, valid_size, test_size): # valid_size is 0 if not given as argument.
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    test_dataset, test_labels = make_arrays(test_size, image_size)

    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    tesize_per_class = test_size // num_classes

    start_v, start_t, start_te = 0, 0, 0
    end_v, end_t, end_te= vsize_per_class, tsize_per_class, tesize_per_class
    end_l = vsize_per_class + tsize_per_class
    end_tst = end_l + tesize_per_class


    for label, pickle_file in enumerate(pickle_files):
        #print(start_v, end_v)
        #print(start_t, end_v)
        name = (pickle_file.split('/')[-1]).split('.')[0]
        try:
            with open(pickle_file, 'rb') as f:
                species_set = pickle.load(f) # set of images from species
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(species_set) # shuffle the data (the "images") in the pickle around


                print("Valid dataset with", name, ". Has ", len(species_set), " images.")
                print("Needs %d images per class" % vsize_per_class)
                print("valid_species is species_set[:%d,:,:]" % vsize_per_class)
                valid_species = species_set[:vsize_per_class, :, :]
                print("valid_dataset[%d:%d,:,:] = valid_species" % (start_v,end_v))
                valid_dataset[start_v:end_v, :, :] = valid_species
                print("valid_labels[%d:%d] = %d" % (start_v,end_v,label))
                valid_labels[start_v:end_v] = label
                start_v += vsize_per_class
                end_v += vsize_per_class


                print("Train dataset with", name, ". Has ", len(species_set), " images")
                print("Needs %d images per class" % tsize_per_class)
                print("train_species is species_set[%d:%d,:,:]" % (vsize_per_class,end_l))
                train_species = species_set[vsize_per_class:end_l, :, :]
                print("train_dataset[%d:%d,:,:] = train_species" % (start_t,end_t))
                train_dataset[start_t:end_t, :, :] = train_species
                print("train_labels[%d:%d] = %d" % (start_t,end_t,label))
                train_labels[start_t:end_t] = label # give label to all images in class
                start_t += tsize_per_class # offset start of class for next iteration
                end_t += tsize_per_class # offset end of class for next iteration

                print("Test dataset with", name, ". Has ", len(species_set), " images")
                print("Needs %d images per class" % tesize_per_class)
                print("test_species is species_set[%d:%d,:,:]" % (end_l, end_te))
                test_species = species_set[end_l:end_tst, :, :]
                print("test_dataset[%d:%d,:,:] = test_species" % (start_te,end_te))
                test_dataset[start_te:end_te, :, :] = test_species
                print("test_labels[%d:%d] = %d" % (start_te,end_te,label))
                test_labels[start_te:end_te] = label # give label to all images in class
                start_te += tesize_per_class # offset start of class for next iteration
                end_te += tesize_per_class # offset end of class for next iteration

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            pass

    return valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels


# In[53]:


train_size = int(total_images * 0.6)
valid_size = int(total_images * 0.2)
test_size = int(total_images * 0.2)
valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels = merge_datasets_all(pickle_subclasses, train_size, valid_size, test_size)


# In[54]:


# create dataset when dataset is not balanced, but want to use entire dataset
def merge_datasets_forced(pickle_files, train_size, valid_size=0): # valid_size is 0 if not given as argument.
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)

    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        #print(start_v, end_v)
        #print(start_t, end_v)
        name = (pickle_file.split('/')[-1]).split('.')[0]
        try:
            with open(pickle_file, 'rb') as f:
                species_set = pickle.load(f) # set of images from species
                if(len(species_set) < (tsize_per_class + vsize_per_class)):
                    # since our dataset is not balanced we need to make sure
                    # we're not taking more images than we have or dimmensions will not match
                    # reset our ends to previous state, calculate new images_per_class, and
                    # calculate new ends

                    end_v -= vsize_per_class
                    end_t -= tsize_per_class

                    tsize_per_class = len(species_set) // 2
                    vsize_per_class = len(species_set) // 2

                    end_v += vsize_per_class
                    end_t += tsize_per_class
                    end_l = vsize_per_class+tsize_per_class



                    # let's shuffle the letters to have random validation and training set
                    np.random.shuffle(species_set) # shuffle the data (the "images") in the pickle around
                    if valid_dataset is not None: # if not testing dataset

                        print("Valid dataset with", name, ". Has ", len(species_set), " images.")
                        print("Needs %d images per class" % vsize_per_class)
                        print("valid_species is species_set[:%d,:,:]" % vsize_per_class)
                        valid_species = species_set[:vsize_per_class, :, :,:]
                        print("valid_dataset[%d:%d,:,:] = valid_species" % (start_v,end_v))
                        valid_dataset[start_v:end_v, :, :,:] = valid_species
                        print("valid_labels[%d:%d] = %d" % (start_v,end_v,label))
                        valid_labels[start_v:end_v] = label

                        # increment our start by how many images we used for this class
                        start_v += vsize_per_class
                        # assume next class will have the required images_per_class
                        end_v += (valid_size // num_classes)

                        # can't reset vsize_per_class here since
                        # the training dataset needs it's current state

                    print("Train dataset with", name, ". Has ", len(species_set), " images")
                    print("Needs %d images per class" % tsize_per_class)
                    print("train_species is species_set[%d:%d,:,:]" % (vsize_per_class,end_l))
                    train_species = species_set[vsize_per_class:end_l, :, :,:]
                    print("train_dataset[%d:%d,:,:] = train_species" % (start_t,end_t))
                    train_dataset[start_t:end_t, :, :,:] = train_species
                    print("train_labels[%d:%d] = %d" % (start_t,end_t,label))
                    train_labels[start_t:end_t] = label # give label to all images in class

                    # increment our start by how many images we used for this class
                    start_t += tsize_per_class
                    # assume next round will have required images_per_class
                    tsize_per_class = train_size // num_classes
                    end_t += tsize_per_class # offset end of class for next iteration
                    vsize_per_class = valid_size // num_classes
                    end_l = vsize_per_class+tsize_per_class


                else: # we have enough images in this class to use our desired imgs_per_class


                    tsize_per_class = train_size // num_classes
                    vsize_per_class = valid_size // num_classes
                    end_l = vsize_per_class+tsize_per_class


                    # let's shuffle the letters to have random validation and training set
                    np.random.shuffle(species_set) # shuffle the data (the "images") in the pickle around
                    if valid_dataset is not None: # if not testing dataset

                        print("Valid dataset with", name, ". Has ", len(species_set), " images.")
                        print("Needs %d images per class" % vsize_per_class)
                        print("valid_species is species_set[:%d,:,:]" % vsize_per_class)
                        valid_species = species_set[:vsize_per_class, :, :,:]
                        print("valid_dataset[%d:%d,:,:] = valid_species" % (start_v,end_v))
                        valid_dataset[start_v:end_v, :, :,:] = valid_species
                        print("valid_labels[%d:%d] = %d" % (start_v,end_v,label))
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class


                    print("Train dataset with", name, ". Has ", len(species_set), " images")
                    print("Needs %d images per class" % tsize_per_class)
                    print("train_species is species_set[%d:%d,:,:]" % (vsize_per_class,end_l))
                    train_species = species_set[vsize_per_class:end_l, :, :,:]
                    print("train_dataset[%d:%d,:,:] = train_species" % (start_t,end_t))
                    train_dataset[start_t:end_t, :, :,:] = train_species
                    print("train_labels[%d:%d] = %d" % (start_t,end_t,label))
                    train_labels[start_t:end_t] = label # give label to all images in class



                    start_t += tsize_per_class # offset start of class for next iteration
                    end_t += tsize_per_class # offset end of class for next iteration

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
        print()

    return valid_dataset, valid_labels, train_dataset, train_labels


# In[55]:


def genLabelMap(pickle_files):
    label_map = {}
    for label, pickle_file in enumerate(pickle_files):
        name = (pickle_file.split('/')[-1]).split('.')[0]
        label_map[label] = name
    return label_map

def sampleCheck(dataset, labels, label_map):
    i = random.randint(1, 5)
    for p_i, img in enumerate(random.sample(range(len(labels)), 5*i)):
        plt.subplot(i, 5, p_i+1)
        plt.axis('off')
        label = labels[img]
        species = label_map[label]
        #print(species)
        title =  species + ' sample:'
        plt.title(title)
        plt.imshow(dataset[img])
        plt.show()


# In[58]:


label_map = genLabelMap(pickle_subclasses)
#sampleCheck(train_dataset, train_labels,label_map)


# ### Output Data

# We output the data in a pickle format, to be used next on the models.

# In[29]:


pickle_file = '../dataset/Shifted_Pickles/augmented_shifted_' + str(num_shifts) + '.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL) # save all out datasets in one pickle
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
