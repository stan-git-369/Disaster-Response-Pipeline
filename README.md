## Disaster Response Pipeline Project
# Project Description

In this project, we will build a model to classify messages that are sent during disasters. Initially we have 36 pre-defined categories, and examples of these categories include Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task. This is also a multi-label classification task, since a message can belong to one or more categories. We will be working with a data set provided by Figure Eight containing real messages that were sent during disaster events.

Finally, this project contains a web app where you can input a message and get classification results.

**Installation:**
This repository was written in HTML and Python , and requires the following Python packages: pandas, numpy,  nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings, re, pickle.

**Project Overview:**
This code is designed to iniate a web app which an emergency operators could exploit during a disaster (e.g. an earthquake or Tsunami), to classify a disaster text messages into several categories which then can be transmited to the responsible entity

The app built to have an ML model to categorize every message received

**File Description:**
* process_data.py: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
* train_classifier.py: This code trains the ML model with the SQL data base
* ETL Pipeline Preparation.ipynb: process_data.py development procces
* ML Pipeline Preparation.ipynb: train_classifier.py. development procces
* data: This folder contains sample messages and categories datasets in csv format.
* app: cointains the run.py to iniate the web app.

**Instructions:**

Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. python run.py

# Acknowledgements
* Udacity for providing an amazing Data Science Nanodegree Program 
* Figure Eight for providing the relevant dataset to train the model

# Screenshots
1. Screenshot 1: App word search Page 
![App word search Page](https://github.com/stan-git-369/Disaster-Response-Pipeline/blob/main/disaster-response-project-main.jpg)

2.


