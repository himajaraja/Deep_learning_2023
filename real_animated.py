# import packages
from tensorflow import keras
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, Activation, Conv2DTranspose, Add
from keras.models import Model


# hyer parameters for model
EPOCHS = 100
BATCH_SIZE = 1




# Getting file list
file_list = os.listdir('real')
# iterating through images
for file_name in file_list:
    file_path = os.path.join('real', file_name)
    # reading images
    image = cv2.imread(file_path)
    # smoothing images
    smoothed = cv2.GaussianBlur(image, (3,3), 0)
    # saving images back
    cv2.imwrite(file_path, smoothed)


# creating image data generator
images_a = ImageDataGenerator(horizontal_flip=True, rescale=1./255).flow_from_directory("real", seed=13, batch_size = BATCH_SIZE, target_size=(256, 256))
# creating image data generator
images_b = ImageDataGenerator(horizontal_flip=True, rescale=1./255).flow_from_directory("animated", seed=13, batch_size = BATCH_SIZE, target_size=(256, 256))



# building discriminator
def discriminator():
    # input layer
    input_x = keras.layers.Input(shape=(256, 256, 3))

    # convolution layer
    x = Conv2D(32, kernel_size = 3, strides = 2)(input_x)
    # activation layer
    # x = keras.layers.LeakyReLU(0.2)(x)
    x = Activation('swish')(x)

    # convolution layer
    x = Conv2D(32, kernel_size = 3, strides = 2)(x)
    # activation layer
    # x = keras.layers.LeakyReLU(0.2)(x)
    x = Activation('swish')(x)

    # convolution layer
    x = Conv2D(32, kernel_size = 3, strides = 2)(x)
    # activation layer
    # x = keras.layers.LeakyReLU(0.2)(x)
    x = Activation('swish')(x)

    # convolution layer
    x = Conv2D(32, kernel_size = 3, strides = 2)(x)
    # activation layer
    # x = keras.layers.LeakyReLU(0.2)(x)
    x = Activation('swish')(x)

    # dropout layer
    x = keras.layers.Dropout(0.5)(x)

    # flatten layer
    x = keras.layers.Flatten()(x)
    # last layer
    x = keras.layers.Dense(1)(x)

    # output layer
    output_x = Activation('sigmoid')(x)

    # returning model
    return keras.Model(input_x, output_x)






# bulding generator
def generator():

    # input layer
    input_x = keras.layers.Input(shape=(256, 256, 3))

    # convolution layer
    x = Conv2D(32, kernel_size = 3, strides=1, padding='same')(input_x)
    x = Activation('relu')(x)

    # convolution layer
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x = Add()([x, x1])

    # convolution layer
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x = Add()([x, x1])

    # convolution layer
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x = Add()([x, x1])


    # convolution layer
    x2 = Conv2D(64, kernel_size = 2, strides = 2)(input_x)
    x2 = Activation('relu')(x2)

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x2)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x2 = Add()([x2, x1])

    # convolution layer
    x3 = Conv2D(128, kernel_size = 2, strides = 2)(x2)
    x3 = Activation('relu')(x3)

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x3)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(128, kernel_size = 3, strides=1, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x3 = Add()([x3, x1])

    # convolution layer
    x3 = Conv2DTranspose(64, kernel_size = 2, strides = 2)(x3)
    x3 = Activation('relu')(x3)

    # Concat layer
    x2 = Add()([x3, x2])

    # convolution layer
    x2 = Conv2DTranspose(32, kernel_size = 2, strides = 2)(x2)
    x2 = Activation('relu')(x2)

    # concat layer
    x = Add()([x, x2])
    x = Conv2D(3, kernel_size = 3, strides=1, padding='same')(x)

    # output layer
    output_x = Activation('relu')(x)

    # returning model
    return keras.Model(input_x, output_x)










# defining discriminators
discriminator_a = discriminator()
discriminator_b = discriminator()


# defining generators
generator_ab = generator()
generator_ba = generator()

# printing models
generator_ab.summary()
discriminator_a.summary()


# compiling discriminator
discriminator_a.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['acc'])
discriminator_a.trainable = False


# compiling discriminator
discriminator_b.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['acc'])
discriminator_b.trainable = False


#defining generator input
input_a = Input(shape=(256, 256, 3))
gen_output_ab = generator_ab(input_a)

# defining disciminator output
disc_output_b = discriminator_b(gen_output_ab)
re_output_ab = generator_ba(gen_output_ab)


# defining generator input
input_b = Input(shape=(256, 256, 3))
gen_output_ba = generator_ba(input_b)

# defining discriminator output
disc_output_a = discriminator_a(gen_output_ba)
re_output_ba = generator_ba(gen_output_ba)


# running generator
gan_ab = keras.Model(inputs=input_a, outputs=[disc_output_b, re_output_ab, gen_output_ab])
gan_ab.compile(optimizer=Adam(learning_rate=0.0001), loss=['binary_crossentropy', 'mae', 'mae'], loss_weights=[0.1, 1, 0.01])


# running generator
gan_ba = keras.Model(inputs=input_b, outputs=[disc_output_a, re_output_ba, gen_output_ba])
gan_ba.compile(optimizer=Adam(learning_rate=0.0001), loss=['binary_crossentropy', 'mae', 'mae'], loss_weights=[0.1, 1, 0.01])







# training
for i in range(0, EPOCHS):
    print('EPOCH : ' + str(i))

    # batching inputs
    images_a_batch = next(images_a)[0]
    images_b_batch = next(images_b)[0]

    # target labels
    target_a_batch = np.ones([BATCH_SIZE, BATCH_SIZE])
    target_b_batch = np.ones([BATCH_SIZE,BATCH_SIZE])

    # fitting generators
    gan_ab.fit(images_a_batch, [target_a_batch, images_a_batch, images_a_batch], batch_size = BATCH_SIZE)
    gan_ba.fit(images_b_batch, [target_b_batch, images_b_batch, images_b_batch], batch_size = BATCH_SIZE)


    # getting gen outputs
    images_b_batch_fake = generator_ab.predict(images_a_batch, batch_size = BATCH_SIZE)
    images_a_batch_fake = generator_ba.predict(images_b_batch, batch_size = BATCH_SIZE)

    # disc inputs
    target_a_batch_fake = np.zeros([len(images_a_batch_fake),BATCH_SIZE])
    target_b_batch_fake = np.zeros([len(images_b_batch_fake),BATCH_SIZE])


    # disc inputs
    images_a_batch_discriminator = np.vstack((images_a_batch, images_a_batch_fake))
    images_b_batch_discriminator = np.vstack((images_b_batch, images_b_batch_fake))
    target_a_batch_discriminator = np.vstack((target_a_batch, target_a_batch_fake))
    target_b_batch_discriminator = np.vstack((target_b_batch, target_b_batch_fake))


    # fitting disc
    discriminator_a.fit(images_a_batch_discriminator, target_a_batch_discriminator, batch_size = BATCH_SIZE)
    discriminator_b.fit(images_b_batch_discriminator, target_b_batch_discriminator, batch_size = BATCH_SIZE)


    # disc inputs
    images_a_batch_discriminator = np.vstack((images_a_batch, images_b_batch))
    images_b_batch_discriminator = np.vstack((images_b_batch, images_a_batch))
    target_a_batch_discriminator = np.vstack((target_a_batch, target_a_batch_fake))
    target_b_batch_discriminator = np.vstack((target_b_batch, target_b_batch_fake))


    # fitting disc
    discriminator_a.fit(images_a_batch_discriminator, target_a_batch_discriminator, batch_size = BATCH_SIZE)
    discriminator_b.fit(images_b_batch_discriminator, target_b_batch_discriminator, batch_size = BATCH_SIZE)





# printing 100 test imaghes
x=0
for image_path in os.listdir("real"):
        x+=1
        if x ==100:
            exit()
        # getting real images
        image = cv2.imread("real" + image_path)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis = 0)
        # predicting target inputs
        image_translated = generator_ab.predict(image)
        image_confidence = discriminator_b.predict(image_translated)
        image_translated = np.squeeze(image_translated)
        # saving animated images
        cv2.imwrite('output/' + str(x)  + '.png', image_translated)

