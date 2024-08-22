import streamlit as st
import pandas as pd
import os




# Import profiling capability
pip install ydata_profiling
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit_pandas_profiling import st_profile_report


# ML Part
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image('https://cdn-icons-png.flaticon.com/512/8637/8637101.png')
    st.title('AutoML APP')
    choice = st.radio('Navigation', ['Upload', 'Profiling', 'ML', 'Download'])
    st.info('This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret. Go, give it a try!')

if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

if choice == "Upload":
    st.title("Upload you data for modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sourcedata.csv', index=None)
        st.dataframe(df)
        pass

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)
    pass

if choice == "ML":
    st.title('Machine Learning runs')
    target = st.selectbox('Select your target column', df.columns)
    if st.button("Train model"):
        setup(df, target=target, train_size=0.8, verbose=False, numeric_imputation='median')
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info('Ths is the ML Model')
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')


if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button("Download the Model", f, "best_model.pkl")
