# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:09:16 2023

@author: Afnan Ali
"""

import json
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import re
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect

from geopy.geocoders import Nominatim
import folium

__locations = None
__data_columns = None
__model = None


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns

    with open(r"https://github.com/AfnanAli6/Airbnb-Price-Prediction/blob/main/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']

    global __model
    if __model is None:
        with open(r'https://github.com/AfnanAli6/Airbnb-Price-Prediction/blob/main/Airbnb_Rental_Price_Prediction_Tuned_XGBoost_Model.pkl', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")



def get_data_columns():
    return __data_columns


  
# creating a function for Prediction

def predict_price(host_response_rate, host_acceptance_rate, host_is_superhost, host_identity_verified, latitude, longitude, accommodates,
                  bedrooms, beds, minimum_nights, maximum_nights, availability_30, availability_365, review_scores_cleanliness,
                  review_scores_checkin, review_scores_communication, review_scores_location, instant_bookable, bathrooms,
                  neighbourhood_group, property_type, response_category, compound_score):
    
    try:
        neighbourhood_group_index = __data_columns.index(neighbourhood_group.lower())
    except:
        neighbourhood_group_index = -1
        
    try:
        property_type_index = __data_columns.index(property_type.lower())
    except:
        property_type_index = -1
    
    try:
        response_category_index = __data_columns.index(response_category.lower())
    except:
        response_category_index = -1
        
    
    x = np.zeros(len(__data_columns))
    
    x[0] = host_response_rate
    x[1] = host_acceptance_rate
    x[2] = host_is_superhost
    x[3] = host_identity_verified 
    x[4] = latitude
    x[5] = longitude 
    x[6] = accommodates       
    x[7] = bedrooms
    x[8] = beds
    x[9] = minimum_nights 
    x[10] = maximum_nights 
    x[11] = availability_30 
    x[12] = availability_365 
    x[13] = review_scores_cleanliness
    x[14] = review_scores_checkin
    x[15] = review_scores_communication 
    x[16] = review_scores_location
    x[17] = instant_bookable
    x[18] = bathrooms   
    x[30] = compound_score
    
    if neighbourhood_group_index >= 0:
        x[neighbourhood_group_index] = 19
        
    if property_type_index >= 0:
        x[property_type_index] = 20
    
    if response_category_index >= 0:
        x[response_category_index] = 21
    
    predicted_log_price = round(__model.predict([x])[0],2)
    reversed_log_price = np.exp(predicted_log_price)  
    return reversed_log_price



def main():

    hide_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """

    st.markdown(hide_style, unsafe_allow_html=True)
    
    try:
        load_saved_artifacts()
    except:
        print("Error loading artifacts!")
    
    #st.set_page_config(page_title='Airbnb Rental Price Predictor')
    #st.title('New York Airbnb Rental Price Predictor')
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://wallpapercave.com/wp/wp10784408.png");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center; color: white;'>Airbnb Rental Price Predictor</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Prediction", "Help!"],
        icons=["search", "info-square"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
        )
    
    
    if selected == "Prediction":
    
        bool_value = ['Yes', 'No']
    #if selected == "":
        #st.markdown("""<style>.big-font{font-size:40px !important;}</style>""", unsafe_allow_html=True)
        
        #st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
        
        
        st.subheader("")
        variable = get_data_columns()
        st.markdown(variable)
        st.markdown("##### Section A: input host information below based on your listing setting")
    
        #st.markdown("# Hello")
        #st.subheader("A) Host Performance Infromation")
    
    
    
        host_identity_verified_val = st.radio('Are you a verified host?', bool_value, horizontal = 1)
        if host_identity_verified_val.strip() == "Yes":
            host_identity_verified_bool = 1
        else:
            host_identity_verified_bool = 0
    
    
        host_is_superhost_val = st.radio('Are you a superhost?', bool_value, horizontal = 1)
        if host_is_superhost_val.strip() == "Yes":
            host_is_superhost_bool = 1
        else:
            host_is_superhost_bool = 0    
    
        response_category_list = ["within a day", "within a few hours", "within an hour"]
        response_category_val = st.selectbox('Select your response category : ', response_category_list)
        
        response_rate_val = st.slider('Set your rate of response (%) :', min_value = 0, max_value = 100, value = 50, step = 2)
        acceptance_rate_val = st.slider('Set your rate of acceptance (%) :', min_value = 0, max_value = 100, value = 50, step = 2)
    
    
    
    
    
    
        st.subheader("")
        st.markdown("##### Section B: input location information below based on your property")

        neighbourhood_group_list = ['bronx', 'brooklyn', 'manhattan', 'queens', 'staten island']
        neighbourhood_group_category_val = st.selectbox('Select the neighbourhood group : ', neighbourhood_group_list)
    
    # Input address from the user
        address = st.text_input("Enter place/building name or nearest landmark :", 'New York City')

    # Convert address to latitude and longitude
        geolocator = Nominatim(user_agent = "streamlit_app.py")
        location = geolocator.geocode(address, timeout=30)
        if location is not None:
            latitude = round(location.latitude, 5)
            longitude = round(location.longitude, 5)
            st.write("Location:", location)
            st.write("Latitude:", latitude)
            st.write("Longitude:", longitude)
        
        else:
            st.write("Not found, please try different landmark!")
            latitude = 40.71273
            longitude = -74.00602
            default_location = 'New York City'
            st.write("Current Location: ", default_location)
            st.write("Latitude:", latitude)
            st.write("Longitude:", longitude)
    
    
    
    
        st.subheader("")
        st.markdown("##### Section C: input accommodation information below based on your offering")
    
    
        property_category_list = ["entire home/apt", "hotel room", "shared room"]
        property_category_val = st.selectbox('Select your property type : ', property_category_list)
    
        accommodates_val = st.number_input('How many people it can accommodate?', min_value = 1, max_value = 20, step = 1)
    
        bedrooms_val = st.number_input('How many bedrooms are there?', min_value = 1, max_value = 10, step = 1)
    
        bathrooms_val = st.number_input('How many baths are there?', min_value = 1., max_value = 7., step = 0.5)
    
        beds_val = st.number_input('How many bed(s) given?', min_value = 1, max_value = 15, step = 1)
    
        instant_bookable_val = st.radio('Is it instant bookable?', bool_value, horizontal = 1)
        if instant_bookable_val.strip() == "Yes":
            instant_bookable_bool = 1
        else:
            instant_bookable_bool = 0
    
        availability_30_val = st.slider('Set unit availability in a month', min_value=1, max_value=30, value=15, step=1)
    
        availability_365_val = st.slider('Set unit availability in a year', min_value=1, max_value=365, value=183, step=1)
    
        min_nights_val = st.number_input('What is the minimum night stay?', min_value=1, max_value=1130, value=1, step=1)
    
        max_nights_val = st.number_input('What is the maximum night stay?', min_value=1, max_value=1130, value=1, step=1)
    
    
    
        st.subheader("")
        st.markdown("##### Section D: input review information below based on your frequent guest feedbacks")
   


# Input text from the user

        text = st.text_area("Enter Text", height=100)
        pattern = re.compile('[^a-zA-Z]+')

        if text.strip() == "":
            st.write("Please enter a review that is often mentioned by guests.")
            compound_score = 0.0
            st.write("Sentiment Polarity Score : ", compound_score)

        else:
            lang = detect(text)
            if lang != 'en':
                st.write("Could not interpret the reviews, please try again.")
                compound_score = 0.0
                st.write("Current Sentiment Polarity Score : ", compound_score)
            else:
                cleaned_text = pattern.sub(' ', text).lower()
                analyzer = SentimentIntensityAnalyzer()
                sentiment_scores = analyzer.polarity_scores(cleaned_text)
                compound_score = sentiment_scores['compound']
                st.write("Sentiment Polarity Score : ", compound_score)
    
    
    
        st.subheader("")
        st.markdown("##### Section E: input guest rating information below based on guest feedbacks")
        st.markdown("###### *you can obtain star rating information from:")
        st.markdown("###### Airbnb host account > go to performance > go to reviews")
        st.write("")
        review_scores_cleanliness = st.slider('Cleanliness (how clean was the place?)', min_value=1., max_value=5., value=2.5, step=0.1)
        review_scores_checkin = st.slider('Check-In (how easy was the process?)', min_value=1., max_value=5., value=2.5, step=0.1)
        review_scores_communication = st.slider('Communication (did the host respond messages promptly?)', min_value=1., max_value=5.,
                                                value=2.5, step=0.1)
        review_scores_location = st.slider('Location (was the guest made aware of safety, transportation, points of interest?)',
                                           min_value=1., max_value=5., value=2.5, step=0.1)    
    
    # code for Prediction
        price = ''
    
        
    # creating a button for Prediction
        if st.button('Predict Rental Price'):
        
            price = predict_price(response_rate_val, acceptance_rate_val, host_is_superhost_bool, host_identity_verified_bool, latitude,
                                  longitude, accommodates_val, bedrooms_val, beds_val, min_nights_val, max_nights_val, availability_30_val,
                                  availability_365_val, review_scores_cleanliness, review_scores_checkin, review_scores_communication,
                                  review_scores_location, instant_bookable_bool, bathrooms_val, neighbourhood_group_category_val,
                                  property_category_val, response_category_val, compound_score)
                              
            rounded_price = round(price, 2)
            body = "The predicted rental price of the unit per night is : $" + str(rounded_price)
        
            st.success(body)
    
    
    if selected == "Help!":
        st.subheader("")
        st.markdown("### What is the purpose of this web application?")
        st.markdown("##### Airbnb Rental Price Predictor web-based application incorporates machine learning and sentiment analysis to assist hosts in determining appropriate rental rates for their property listings.")
        st.subheader("")
        st.markdown("### What are the benefits of using this application?")
        st.markdown("##### Some of the benefits of using this application are, maximizing business revenue for Airbnb and its hosts, saving third party valuation fee for determining rental rates, as well as eliminating market research time needed for pricing listings effectively.")        
        st.subheader("")
        st.markdown("### How to use this web application?")
        st.markdown("##### As an Airbnb host, you will need to log into your respective Airbnb account and provide the correct user inputs for the following sections such as 'Host Info', 'Location Info', 'Accommodation Info', 'Review Info', and 'Rating Info'. Then, rental price of the property will be predicted based on the user input given.")
        st.subheader("")
    
if __name__ == '__main__':
    main()
