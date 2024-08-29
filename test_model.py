import keras
from keras.preprocessing import image
reconstructed_model_soybean_leaf = keras.models.load_model("models/leaf.soybean.mobilenetv3large.keras")


import numpy as np

def predict_soybean_leaf_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    classes = reconstructed_model_soybean_leaf.predict(x)
    class_labels = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
    return class_labels[np.argmax(classes)]

print(predict_soybean_leaf_disease("./images/Soybean/Soybean_Caterpillar/caterpillar (1).jpg"))

"""img = image.load_img("./images/Soybean/Soybean_Caterpillar/caterpillar (1).jpg", target_size=(224, 224))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)
classes = reconstructed_model.predict(y)
class_labels = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
print(class_labels[np.argmax(classes)])"""