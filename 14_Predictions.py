#       streamlit run eda_app.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
from pycaret.regression import *
from scipy.stats import percentileofscore
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 200
import plotly.subplots as sp
import plotly.express as px
import plotly.io as pio
import io
import streamlit as st
import os
import pickle
import joblib
import sys


from PIL import Image
import streamlit as st

# You can always call this function where ever you want


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


def train_best_model(
    X_features1,
    X_features2,
    multi_model_results,
    all_models_dict,
    preprocessed_df,
    predict_col,
):
    X_features = []
    for feature, selected in X_features1.items():
        if selected:
            X_features.append(feature)
    for feature, selected in X_features2.items():
        if selected:
            X_features.append(feature)

    # create model of best result
    multi = multi_model_results
    best_model = multi.iloc[0]["Model"]
    model_code = all_models_dict[best_model]
    df_regress = preprocessed_df[X_features]

    reg2 = setup(
        data=df_regress,
        target=predict_col,
        session_id=123,
        verbose=False,
        low_variance_threshold=0.08,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.3,
    )
    model2 = create_model(
        model_code, fold=10, round=4, cross_validation=True, verbose=False
    )

    return model2


st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def check_password():
    """Returns `True` if the user had the correct password."""

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
        st.error("😕 Password incorrect. Email dan9232@nyu.edu to request access.")
        return False
    else:
        # Password correct.
        return True


if check_password():

    my_logo = add_logo(
        logo_path="PropertizeAI-vector-house-trend-white.png", width=180, height=200
    )
    st.sidebar.image(my_logo)

    @st.cache_data
    def read_data(filename):
        df = pd.read_csv(filename)
        return df

    st.sidebar.markdown("#")
    preprocessed_df = read_data("streamlitPredictionsData.csv")
    st.sidebar.header("-Model Predictions Demo-")

    sys.path.append(os.path.abspath(os.path.join("..")))

    @st.cache_data
    def load_ML_model(model_name):
        # with open(model_name, 'rb') as file:
        #     model = pickle.load(file)
        model = load_model(model_name)
        return model

    def predict_saleLikelihood(modelSL, pred_dict):
        predictors = modelSL.feature_names_in_[:-1]
        pred_vector = [pred_dict[p] for p in predictors]
        pred_vector = pd.DataFrame(
            [pred_vector], columns=modelSL.feature_names_in_[:-1]
        )
        prediction = predict_model(modelSL, data=pred_vector)["prediction_label"][0]
        # prediction = modelSL.predict(pred_vector)[0]
        pred_dict["saleLikelihood"] = prediction
        return pred_dict

    def predict_avgSchoolRating(modelASR, pred_dict):
        predictors = modelASR.feature_names_in_[:-1]
        pred_vector = [pred_dict[p] for p in predictors]
        pred_vector = pd.DataFrame(
            [pred_vector], columns=modelASR.feature_names_in_[:-1]
        )
        prediction = predict_model(modelASR, data=pred_vector)["prediction_label"][0]
        # prediction = modelASR.predict(pred_vector)[0]
        pred_dict["avgSchoolRating"] = prediction
        return pred_dict

    def valuateProperty(modelEval, predict_vec_pl2):
        predictors = modelEval.feature_names_in_[:-1]
        pred_vector = [predict_vec_pl2[p] for p in predictors]
        pred_vector = pd.DataFrame(
            [pred_vector], columns=modelEval.feature_names_in_[:-1]
        )
        prediction = predict_model(modelEval, data=pred_vector)["prediction_label"][0]
        # prediction = modelEval.predict(pred_vector)[0]
        predict_vec_pl2["valuationPropertizeAI"] = round(prediction)
        return predict_vec_pl2

    # ['valuateProperty_model', 'saleLikelihood_model_No_HT', 'avgSchoolRating_model']

    st.markdown("# PropertizeAI ML Model Predictions")

    st.markdown("")
    st.sidebar.markdown("##")
    st.sidebar.markdown(
        "## 1. Property Valuation Model: Utilizing a Gradient Boosting Regressor, this model estimates a property's market value based on its features, providing users with an accurate, data-driven valuation for informed decision-making."
    )
    st.sidebar.markdown("##")
    st.sidebar.markdown(
        "## 2. Sale Likelihood Model: This innovative model employs a Linear Regression algorithm to assess the probability of a property selling quickly based on its characteristics, empowering users to optimize their strategy for a more efficient and profitable sale."
    )
    st.markdown("##")

    modelSL = load_ML_model("saleLikelihood_model_No_HT")
    modelASR = load_ML_model("avgSchoolRating_model")
    modelEval = load_ML_model("valuateProperty_model")

    eval_params = [
        "valuationPropertizeAI",
        "saleLikelihood",
        "daysOnMarket",
        "hoaFee",
        "yearBuilt",
        "bedrooms",
        "bathroomsFloat",
        "avgSchoolRating",
        "zipcode",
        "latitude",
        "longitude",
    ]

    with st.form(key="model_pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Enter the values for each feature in the form below")
            daysOnMarket = st.slider(
                "Days on Market", min_value=0, max_value=1000, value=11, step=5
            )
            hoaFee = st.number_input("HOA Fee", min_value=0, max_value=1000, step=300)
            yearBuilt = st.number_input(
                "Year Built", min_value=0, max_value=2021, value=2010, step=1
            )
            bedrooms = st.number_input(
                "Bedrooms", min_value=0, max_value=10, value=4, step=1
            )

        with col2:
            st.header("")
            st.header("")
            st.write("")
            bathroomsFloat = st.number_input(
                "Bathrooms", min_value=0, max_value=10, value=4, step=1
            )
            zipcode = st.number_input(
                "Zipcode", min_value=0, max_value=100000, value=10003, step=1
            )
            latitude = st.text_input(
                "Latitude", placeholder="40.72435", value="40.72435"
            )
            longitude = st.text_input(
                "Longitude", placeholder="-73.98559", value="-73.98559"
            )
        submit_button = st.form_submit_button(label="Generate Property Predictions")

    st.markdown("##")

    if submit_button:
        predict_vector = [
            None,
            None,
            daysOnMarket,
            hoaFee,
            yearBuilt,
            bedrooms,
            bathroomsFloat,
            None,
            zipcode,
            float(latitude),
            float(longitude),
        ]
        predict_dict = dict(zip(eval_params, predict_vector))
        predict_dict = predict_saleLikelihood(modelSL, predict_dict)
        predict_dict = predict_avgSchoolRating(modelASR, predict_dict)
        predict_dict = valuateProperty(modelEval, predict_dict)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.header("Probability of Quick Sale")
            col11, col22, col33 = st.columns(3)
            with col22:
                st.header("")
                st.header("")
                st.header("")
                st.subheader(round(predict_dict["saleLikelihood"], 3))
        with col2:
            fig = px.histogram(
                preprocessed_df,
                x="saleLikelihood",
                nbins=100,
                title="Predicted  Sale Likelihood Distribution",
            )
            fig.add_vline(
                predict_dict["saleLikelihood"],
                line_width=3,
                line_dash="dash",
                line_color="red",
            )
            fig.update_layout(
                template="plotly_dark",
                title_x=0.3,
                autosize=False,
                width=1200,
                height=700,
                title_font_size=30,
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
            )
            # increase font size of title and axis

            st.plotly_chart(fig)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.header("PropertizeAI Valuation")
            col11, col22, col33 = st.columns(3)
            with col22:
                st.header("")
                st.header("")
                st.header("")
                percentile_of_3 = percentileofscore(
                    preprocessed_df["zestimatePriceAvg"],
                    predict_dict["valuationPropertizeAI"],
                )
                lowEstimate = np.percentile(
                    preprocessed_df["zestimatePriceAvg"], percentile_of_3 - 2
                )
                highEstimate = np.percentile(
                    preprocessed_df["zestimatePriceAvg"], percentile_of_3 + 2
                )
                st.subheader((round(lowEstimate, 3)))
                st.subheader(f"Percentile: {percentile_of_3}%")
                st.subheader(
                    "${:,.2f}".format(round(predict_dict["valuationPropertizeAI"], 3))
                )
        with col2:
            fig = px.histogram(
                preprocessed_df,
                x="zestimatePriceAvg",
                nbins=1000,
                title="Predicted Valuation Distribution",
            )

            fig.add_vline(
                x=predict_dict["valuationPropertizeAI"],
                line_width=3,
                line_dash="dash",
                line_color="red",
            )
            # make the plot take up the whole plotly area
            fig.update_layout(
                template="plotly_dark",
                autosize=False,
                title_x=0.3,
                width=1200,
                height=700,
                title_font_size=30,
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                margin=dict(l=50, r=50, b=100, t=100, pad=4),
            )

            st.plotly_chart(fig)
