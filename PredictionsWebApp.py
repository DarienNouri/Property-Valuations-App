#  streamlit run eda_app.py
import pandas as pd
#from pycaret.regression import load_model, predict_model 
from scipy.stats import percentileofscore
import numpy as np
import plotly.express as px
import streamlit as st
import os
import sys
from PIL import Image
import streamlit as st
import joblib
import censusgeocode as cg

sys.path.append(os.path.abspath(os.path.join('..')))
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

IS_LOGIN_PAGE = False

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def check_password():
    """Returns `True` if the user had the correct password."""

    global IS_LOGIN_PAGE
    if not IS_LOGIN_PAGE:
        return True
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("# PropertizeAI ML Property Valuations Demo")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Incorrect Password. Email dan9232@nyu.edu to request access.")
        return False
    else:
        return True
        
if check_password():       
    my_logo = add_logo(logo_path="PropertizeAI-vector-house-trend-white.png", width=180, height=200)
    st.sidebar.image(my_logo)

    @st.cache_data
    def read_data(filename):
        df = pd.read_csv(filename)
        df.rename(columns={'zestimatePriceAvg': 'Property Value'}, inplace=True)
        return df
    
    st.sidebar.markdown('#')
    preprocessed_df = read_data('streamlitPredictionsData.csv')
    st.sidebar.header("-Model Predictions Demo-")
        
    @st.cache_data
    def load_ML_model(model_name):
        # with open(model_name, 'rb') as file:
        #     model = pickle.load(file)
        model = joblib.load(model_name)
        #model = load_model(model_name)
        return model

    def requestCencusData(street, city, state):
        cdata = cg.address(street, city, state)
        if cdata:
            zipcode = cdata[0].get('addressComponents').get('zip')
            lon = cdata[0].get('coordinates').get('x')
            lat = cdata[0].get('coordinates').get('y')
            return [zipcode, lon, lat]
        else:
            return None, None, None
    
    def predict_saleLikelihood(modelSL, pred_dict):
        predictors = modelSL.feature_name_[:]
        pred_vector = [pred_dict[p] for p in predictors]
        pred_vector = pd.DataFrame([pred_vector], columns=modelSL.feature_name_[:])
        #prediction = predict_model(modelSL, data=pred_vector)['prediction_label'][0]
        modelSL._n_classes = 2
        prediction = modelSL.predict(pred_vector)[0]
       #prediction = modelSL.predict(pred_vector)[0]
        pred_dict['saleLikelihood'] = prediction
        return pred_dict

    def predict_avgSchoolRating(modelASR, pred_dict):
        predictors = modelASR.feature_name_[:]
        pred_vector = [pred_dict[p] for p in predictors]
        pred_vector = pd.DataFrame([pred_vector], columns=modelASR.feature_name_[:])
        #prediction = predict_model(modelASR, data=pred_vector)['prediction_label'][0]
        prediction = modelASR.predict(pred_vector)[0]
        pred_dict['avgSchoolRating'] = prediction
        return pred_dict


    def valuateProperty(modelEval, predict_vec_pl2):
        predictors = modelEval.feature_name_[:]
        pred_vector = [predict_vec_pl2[p] for p in predictors]
        pred_vector = pd.DataFrame([pred_vector], columns=modelEval.feature_name_[:])
        #prediction = predict_model(modelEval, data=pred_vector)['prediction_label'][0]
        modelEval._n_classes = 2
        prediction = modelEval.predict(pred_vector)[0]
        predict_vec_pl2['valuationPropertizeAI'] = round(prediction)
        return predict_vec_pl2

    st.markdown("# PropertizeAI ML Model Predictions")
    st.markdown("")
    st.sidebar.markdown("##")
    st.sidebar.markdown("## 1. Property Valuation Model: Utilizing a Gradient Boosting Regressor, this model estimates a property's market value based on its features, providing users with an accurate, data-driven valuation for informed decision-making.")
    st.sidebar.markdown("##")
    st.sidebar.markdown("## 2. Sale Likelihood Model: This innovative model employs a Linear Regression algorithm to assess the probability of a property selling quickly based on its characteristics, empowering users to optimize their strategy for a more efficient and profitable sale.")
    st.markdown("##")

    modelSL = load_ML_model('saleLikelihood_rawModel.pkl')
    modelASR = load_ML_model('avgSchoolRating_rawModel.pkl')
    modelEval = load_ML_model('valuateProperty_rawModel.pkl')

    eval_params = ['valuationPropertizeAI',
                'saleLikelihood', 'daysOnMarket',
                    'hoaFee', 'yearBuilt', 'bedrooms',
                    'bathroomsFloat', 'avgSchoolRating',
                    'zipcode', 'latitude', 'longitude']


    with st.form(key='model_pred_form'):
        col1, col2 = st.columns(2)
        with col1:
            daysOnMarket = st.slider('Days on Market', min_value=0, max_value=1000, value=25, step=5)
            bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=4, step=1)
            bathroomsFloat = st.number_input('Bathrooms', min_value=0, max_value=10, value=4, step=1)
            yearBuilt = st.number_input('Year Built', min_value=0, max_value=2021, value=2010, step=1)

        with col2:
            #st.header("")
            # st.header("")
            st.write("")
            hoaFee = st.number_input('HOA Fee', min_value=0, max_value=10000,  step=250)
            streetAddress = st.text_input('Street Address', placeholder='123 Main St', value='6 W 14th St')
            city = st.text_input('City', placeholder='New York', value='New York')
            state = st.text_input('State', placeholder='NY', value='NY')
            # latitude = st.text_input('Latitude', placeholder='40.72435', value='40.72435')
            # longitude = st.text_input('Longitude', placeholder='-73.98559', value='-73.98559')
        submit_button = st.form_submit_button(label='Generate Property Predictions')

    st.markdown('##')
    
    if submit_button: 
        zipcode, longitude, latitude = requestCencusData(streetAddress, city, state)
        if zipcode is None:
            st.error('Invalid Address')

    if submit_button and zipcode is not None:
        predict_vector = [None, None, daysOnMarket, hoaFee, yearBuilt, bedrooms, bathroomsFloat, None, int(zipcode), latitude, longitude]
        predict_dict = dict(zip(eval_params, predict_vector))
        predict_dict = predict_saleLikelihood(modelSL, predict_dict)
        predict_dict = predict_avgSchoolRating(modelASR, predict_dict)
        predict_dict = valuateProperty(modelEval, predict_dict)
        

        col1, col2 = st.columns([1,3])
        with col1:
            st.header("Probability of Quick Sale")
            col11, col22, col33 = st.columns(3)
            with col22:
                st.header("")
                st.header("")
                st.header("")
                st.subheader(round(predict_dict['saleLikelihood'],3))
        with col2:
            fig = px.histogram(preprocessed_df, x="saleLikelihood", nbins=100, title="Predicted  Sale Likelihood Distribution")
            fig.add_vline(predict_dict['saleLikelihood'], line_width=3, line_dash="dash", line_color="red")
            fig.update_layout(
                template='plotly_dark',
                title_x=0.3,
                autosize=False,
                width=1200,
                height=700,
                title_font_size=30,
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                ), 
            )
            st.plotly_chart(fig)

        col1, col2 = st.columns([1,3])
        with col1:
            st.header("PropertizeAI Valuation")
            st.header("")
            st.header("")
            st.header("")
            percentile_of_3 = percentileofscore(preprocessed_df['Property Value'], predict_dict['valuationPropertizeAI'])
            lowEstimate = round(np.percentile(preprocessed_df['Property Value'], percentile_of_3 - .5), -4)
            highEstimate = round(np.percentile(preprocessed_df['Property Value'], percentile_of_3 + .5), -4)
            st.markdown("### Valuation Range:")
            st.write("")
            lowEstFormat = "Low: ${:,.2f}".format(lowEstimate)
            highEstFormat = "High: ${:,.2f}".format(highEstimate)
            st.markdown("### " + lowEstFormat)
            st.markdown("### " + highEstFormat)

        with col2:
            fig = px.histogram(preprocessed_df, x="Property Value", nbins=1000, title="Predicted Valuation Distribution")
            fig.add_vline(x=predict_dict['valuationPropertizeAI'], line_width=3, line_dash="dash", line_color="red")
            fig.update_layout(
                template='plotly_dark',
                autosize=False,
                title_x=0.3,
                width=1200,
                height=700,
                title_font_size=30,
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                # change x axis title to Price
                xaxis_title="Price",
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                ),
            )
            st.plotly_chart(fig)


