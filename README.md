![alt text](https://github.com/MrDomian/Heart-Disease-Risk-Prediction/blob/main/Data/Risk_of_heart_disease_banner.jpg)
# Heart Disease Risk Prediction

Welcome to the "Heart Disease Risk Prediction:" project repository! This project is dedicated to applying data analysis and machine learning techniques to identify the risk of heart disease based on available patient features. The primary aim of this project is to provide insights that can be beneficial in the medical field, supporting both diagnosis and clinical decision-making processes.

## Data source

The project uses the heart disease dataset available on kaggle, provided by John Smith. This dataset compiles data from various institutions, making it a comprehensive resource for our analysis.

You can find the dataset here: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Motivation

In today's world, with the increasing availability of vast amounts of medical data and advancements in machine learning, there is an opportunity to harness data-driven technologies for predicting the risk of heart diseases. This possibility highlights the immense potential that data analysis holds within the field of medicine.

The first article I found, titled “Artificial intelligence-based detection of aortic stenosis from chest radiographs” published in the European Heart Journal - Digital Health, discusses the use of artificial intelligence to detect aortic stenosis from chest radiographs. The study aimed to develop models to detect aortic stenosis (AS) from chest radiographs - one of the most basic imaging tests - with artificial intelligence. The results showed that deep learning models using chest radiographs have the potential to differentiate between radiographs of patients with and without AS. This article is an example of how artificial intelligence-based technologies are becoming more sophisticated and can be used to identify early signs of heart diseases and predict their occurrence.

Source of the article: https://academic.oup.com/eurheartj/article/43/8/716/6472699

Heart diseases continue to be a significant public health concern worldwide. Thanks to the growing availability of medical data and progress in machine learning, there is a potential for utilizing data-based techniques to predict and comprehend the risk of heart diseases. This project aims to contribute to this vital domain by investigating pertinent attributes, constructing predictive models, and sharing findings with the community.

## The structure of key files

- **Data preparation.ipynb**: This notebook covers data preprocessing operations such as data cleaning, transformation, and normalization. The goal is to make the data suitable for subsequent analysis and modeling.
- **Data analysis.ipynb**: This notebook provides a detailed analysis of the dataset, including visualizations, descriptive statistics, and insights to help understand features and relationships in the data.
- **Feature engineering.ipynb**: Here, you'll find the implementation of various methods and strategies for enhancing data features. This includes transformations, feature creation, scaling, dimensionality reduction, and feature selection.
- **Machine learning.ipynb**: In this notebook, we delve into the realm of machine learning. We explore various algorithms for heart disease risk detection, evaluate their performance, and extract relevant information for each model.
- **Heart disease folder**: Contains raw datasets, additional information about them, and other relevant files.
- **Data folder**: Contains files with results and other less important data.
- **SQL folder**: Contains exercises and queries for learning SQL using PostgreSQL.
- **Sav models folder**: Contains exported models for use in the streamlit application.
- **heart_disease_risk.csv**: Processed CSV file containing heart disease data, prepared for analysis and modeling.
- **processed.cleveland.data**: File containing the original unprocessed heart disease data, located in the "Heart disease" folder.
- **streamlit_app.py**: Implements the saved models for the streamlit application.

## Installation

To replicate or explore this project, please make sure you have the required Python packages installed. You can install them using the provided requirements.txt file:
- **pip install -r requirements.txt**

## Streamlit Page

As part of this project, an interactive web page has been created using the Streamlit tool. This page incorporates implemented machine learning models for predicting the risk of heart disease. It enables users to input relevant values and parameters, upon which predictions concerning the presence or absence of heart disease are generated.

Page Features:
- **Data Input Form**: Users can input required values, such as age, blood pressure, cholesterol level, etc., which are necessary for prediction.
- **Prediction**: The page employs implemented machine learning models to predict whether an individual is at risk of heart disease or not, based on the provided data.
- **Aggregated Chart**: The page generates an aggregated bar chart that illustrates the distribution of predictions across risk categories. This chart offers a swift overview of how algorithms assess the presence or absence of heart disease.

Web Page Link: **https://heart-disease.streamlit.app**

![alt text](https://github.com/MrDomian/Heart-Disease-Risk-Prediction/blob/main/Data/screen_streamlit_page.png)

## Usage

Feel free to explore the notebooks provided in this repository to gain insights into different aspects of the project, from data preparation to machine learning model evaluation. Your contributions to this project are more than welcome! If you have any suggestions, ideas, or feedback, please don't hesitate to open an issue or submit a pull request. Happy exploring!

## Principal dataset researchers

The heart disease dataset draws contributions from renowned medical institutions:
1. Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
