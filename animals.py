# Импорт необходимых библиотек
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузка предобученной модели VGG16
base_model = VGG16(weights='imagenet', include_top=True)

# Загрузка и предобработка изображения
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Классификация объекта
preds = base_model.predict(x)
print('Результат классификации объекта:', decode_predictions(preds, top=1)[0])
