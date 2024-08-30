"""import keras
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

print(predict_soybean_leaf_disease("./images/Soybean/Soybean_Caterpillar/caterpillar (1).jpg"))"""
import os

"""img = image.load_img("./images/Soybean/Soybean_Caterpillar/caterpillar (1).jpg", target_size=(224, 224))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)
classes = reconstructed_model.predict(y)
class_labels = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
print(class_labels[np.argmax(classes)])"""


import arrow
import requests
import geocoder

lat, lng = geocoder.arcgis("Concord, NC").latlng
print(lat, lng)
# Get first hour of today
start = arrow.now().floor('day')

# Get last hour of today
end = arrow.now().ceil('day')

response = requests.get(
  'https://api.stormglass.io/v2/bio/point',
  params={
    'lat': lat,
    'lng': lng,
    'params': 'soilMoisture,soilTemperature',
    'start': start.to('UTC').timestamp(),  # Convert to UTC timestamp
    'end': end.to('UTC').timestamp()  # Convert to UTC timestamp
  },
  headers={
    'Authorization': os.getenv('STORMGLASS_API')
  }
)

# Do something with response data.
json_data = response.json()
import pprint
pprint.pprint(json_data['hours'][-1]["soilMoisture"]["noaa"]*100)
print(start.to('UTC'))