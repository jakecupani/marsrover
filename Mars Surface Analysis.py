import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Root File Path
rootfp = "C:/Users/Jake/Documents/GitHub/marsterrain/data/"

# Reference Dataframe
ref_df = pd.read_csv("msl_synset_words-indexed.txt",header=None,delimiter='\d+',usecols=[1],engine='python')
ref_df = ref_df.reset_index()
ref_df.columns = ['class','label']

# Set Batch and Image Size
image_size = (256, 256)
batch_size = 32

# Initialize Train Data Directory
data_dir_train = pathlib.Path(rootfp + "train")
data_dir_val = pathlib.Path(rootfp + "validation")

# Generate Datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_val,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

