import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from keras.models import load_model
from tensorflow.keras import preprocessing
import time


st.title('Weed Classifier Using CNN')

st.markdown("Prediksi Jenis Gulma")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

    col1, col2, col3 , col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        class_btn = st.button("Classify") 
 
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
        
                predictions = predict(image)
                time.sleep(1)
                st.success(predictions)


def predict(image):
    classifier_model = "weed_model_classification.h5"
      
    model = load_model(classifier_model)
      
    test_image = image.resize((300, 300))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['alligatorweed', 'asiatic-smartweed', 'barnyard-grass', 'bidens', 'billygoat-weed', 'black-nightshade', 'ceylon-spinach', 'chinese-knotweed', 'clidemia-hirta',
               'cocklebur', 'common-dayflower','crabgrass', 'dicranopteris-linearis', 'field-thistle', 'goosefoots', 'green-foxtail', 'horseweed', 'indian', 'mock-strawberry',
               'pigweed', 'plantian', 'purslane', 'sedge', 'shepherd-purse', 'velvetleaf', 'viola', 'white-smart-weed']

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    result = f"{class_names[np.argmax(scores)]} with a { (1000 * np.max(scores)).round(2) } % confidence." 
    return result

if __name__ == "__main__":
    main()