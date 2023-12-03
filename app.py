import streamlit as st
import numpy as np
import tensorflow as tf
import time
import joblib

st.set_page_config(page_title='Next Word',page_icon="⏭️")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image("Images/title.png")
text_input = st.text_input("Please Provide a Word", "Police")

model = tf.keras.models.load_model("Saved Model/TF_Model_500.h5")
tokenizer = joblib.load("Saved Model/tokenizer")

def predict_next_word(text):
    for i in range(6):
      tokenize_text = tokenizer.texts_to_sequences([text])[0]
      padded_text = tf.keras.utils.pad_sequences([tokenize_text], maxlen=50)
      similar_position = np.argmax(model.predict(padded_text))

      for word,index in tokenizer.word_index.items():
        if index == similar_position:
          text = text + " " + word
          if st.button(word, use_container_width=True):
            st.write(text)
          time.sleep(0.5)

if st.button("Click to Suggest Next Word", use_container_width=True):   
    prediction = predict_next_word(text_input)
    st.balloons()

    
