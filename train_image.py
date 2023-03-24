# Importing all necessary libraries
import keras
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
individual_train_image = 'data/train/clean/Clean.png'
nb_train_samples = 192
nb_validation_samples = 20
epochs = 30
batch_size = 16

# Create function which will tweak image to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

img = keras.utils.load_img(individual_train_image)

x = keras.utils.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix='messy',
                          save_format='png'):
    i += 1
    if i > 96:
        break
