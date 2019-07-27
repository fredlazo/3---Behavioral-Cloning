import cv2
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

# Define path for input data
data_dir = 'data/data/'
csv_file = data_dir + 'driving_log.csv'

# Import data from the csv file, omit the first line
lines = []
with open( csv_file ) as input:
    reader = csv.reader( input )
    for line in reader:
        lines.append( line )
lines = lines[1:]

# 80% training
train_samples, validation_samples = train_test_split( lines, test_size=0.2 )

# Generator for fit data
def generator( samples, batch_size=32 ):
    n_samples = len( samples )

    # Loop forever so the generator never ends
    while 1:
        sklearn.utils.shuffle( samples )

        # Loop through batches
        for offset in range( 0, n_samples, batch_size ):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]
               
                path_center = 'data/data/IMG/' + filename_center
                path_left = 'data/data/IMG/' + filename_left
                path_right = 'data/data/IMG/' + filename_right
                
                # Use mpimg.imread to train on data in RGB
                image_center = mpimg.imread( path_center )
                image_left = mpimg.imread( path_left )
                image_right = mpimg.imread( path_right )
                # Flip center image
                flipped_image = np.copy( np.fliplr( image_center ) )
                
                images.append( image_center )
                images.append( image_left )
                images.append( image_right )
                images.append( flipped_image )
                
                # Correction angle added/subtracted to create driving angle for the left & right images
                correction = 0.065
                angle_center = float( batch_sample[3] )
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                flipped_angle = -angle_center
                
                angles.append( angle_center )
                angles.append( angle_left ) 
                angles.append( angle_right )
                angles.append( flipped_angle )

            X_train = np.array( images )
            y_train = np.array( angles )
           
            yield sklearn.utils.shuffle( X_train, y_train )

print( len( train_samples ) )
print( len( validation_samples ) )

# Assign the generator to the training samples and validation samples
train_generator = generator( train_samples, batch_size=32 )
validation_generator = generator( validation_samples, batch_size=32 )

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Crop irrelevant image features
model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
#Normalize
model.add( Lambda( lambda x: x/255. - 0.5 ) )
# Nvidia Network
model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Flatten() )
model.add( Dense( 100 ) )
# Dropout to combat overfitting
model.add(Dropout(0.5))
model.add( Dense( 50 ) )
model.add( Dense( 10 ) )
model.add( Dense( 1 ) )

model.compile( loss='mse', optimizer='adam' )

# Number of times generators called
training_steps = np.ceil( len( train_samples )/32 ).astype( np.int32 )
validation_steps = np.ceil( len( validation_samples )/32 ).astype( np.int32 )

model.fit_generator( train_generator, \
    steps_per_epoch = training_steps, \
    epochs=5, \
    verbose=1, \
    callbacks=None, 
    validation_data=validation_generator, \
    validation_steps=validation_steps, \
    class_weight=None, \
    max_q_size=10, \
    workers=1, \
    pickle_safe=False, \
    initial_epoch=0 )

model.save( 'model.h5' )
