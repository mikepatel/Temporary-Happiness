"""
"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


################################################################################
# Filepaths
DATA_DIR = os.path.join(os.getcwd(), "data")
CSV_FILEPATH = os.path.join(DATA_DIR, "data.csv")
PLOT_FILEPATH = os.path.join(os.getcwd(), "training_accuracy")


################################################################################
NUM_EPOCHS = 5
BATCH_SIZE = 16
NUM_RNN_UNITS = 128
MAX_WORDS = 5000  # limit data to top x words
MAX_SEQ_LENGTH = 100  #
EMBEDDING_DIM = 100  #
