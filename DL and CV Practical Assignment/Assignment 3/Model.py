import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

unique_breeds = []

for folders in os.listdir("Images"):
    breed = "".join(folders.split("_"[1:]))
    unique_breeds.append(breed)
unique_breeds = np.array(sorted(unique_breeds))
len(unique_breeds),unique_breeds[:10]