import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import base64

# Loading Trained Model
MyModel = pickle.load(open('forest-cover.pkl', 'rb'))

# Creating web app
st.title('Forest Cover Prediction')

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background('Background.jpg')

user_input = st.text_input(
    "**Enter all cover Type Features separated by comma**",
    key="user_input",
    help="Example: 11,2612,201,4,180,51,735,218,243,161,6222,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
)

st.markdown(
    """
    <style>
    .stTextInput label {
        font-size: 20px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if user_input:
    # Split the string into a list of strings
    user_input_list = user_input.split(',')

    # Convert the list of strings to a list of integers
    features = np.array(user_input_list, dtype='int32')

    # Making a prediction
    prediction = MyModel.predict([features])

    # Display the prediction
    st.write(f'The predicted forest cover type is: {prediction[0]}')

    # Creating a dictionary to map the cover type to the actual cover type
    cover_type = {
        1: {'name': 'Spruce/Fir', 'image': 'spruce-fur-1.jpg'},
        2: {'name': 'Lodgepole Pine', 'image': 'lodge-pole-pine-2.jpg'},
        3: {'name': 'Ponderosa Pine', 'image': 'ponderosa-pine-3.jpg'},
        4: {'name': 'Cottonwood/Willow', 'image': 'cottonwood-willow-4.jpg'},
        5: {'name': 'Aspen', 'image': 'aspen-5.jpg'},
        6: {'name': 'Douglas-fir', 'image': 'douglus-fir-6.jpg'},
        7: {'name': 'Krummholz', 'image': 'krummholz-7.jpg'}
    }

    cover_type_info = cover_type[prediction[0]]

    if cover_type_info is not None:
        forest_name = cover_type_info['name']
        image_path = cover_type_info['image']

        col1, col2 = st.columns([1, 3])
        with col1:
            st.write("This is the Predicted Forest Cover Type ")
            st.write(f"<h1>{forest_name}</h1>", unsafe_allow_html=True)
        
        with col2:
            final_img = Image.open(image_path)
            st.image(final_img, caption=forest_name, use_column_width=True)
