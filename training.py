import tensorflow as tf 
import keras


# Parameters
run_id = 'run1'
epochs = 3
no_validation_images = 4
no_training_images = 5
batch_size = 1
learning_rate = 0.001
no_batches = no_training_images // batch_size

