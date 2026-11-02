Crop Yield Forecast

A machine learning project that predicts crop yield from agricultural and environmental data, helping farmers and planners make better decisions.

B.Tech Mini Project (2026-27) | Domain: Machine Learning and Predictive Analytics

Table of Contents
About the Project
Features
Tech Stack
Dataset
Project Structure
Installation
Usage
Results
Future Scope
Author
About the Project

Crop yield depends on many factors such as rainfall, temperature, soil type, fertilizer use and crop variety. This project trains machine learning models on historical data to forecast the expected yield for a given crop and set of conditions.

Objectives

Clean and prepare the agricultural dataset
Train and compare machine learning models
Predict crop yield for new input values
Evaluate models using standard error metrics
Features
Data preprocessing (missing values, encoding, scaling)
Exploratory data analysis with charts
Multiple regression models compared on the same data
Prediction on new user input
Model performance evaluation (R2 score, MAE, RMSE)
Tech Stack
Language: Python 3.x
Libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn
Tools: VS Code, Git, GitHub
Dataset
Source: (add the dataset name and link, for example Kaggle)
Records: (number of rows)
Key columns: (for example crop, season, state, area, rainfall, fertilizer, pesticide, yield)
Project Structure
Mini_Project/
|-- code.py            # Main script (training and prediction)
|-- dataset.csv        # Dataset (rename to match your file)
|-- requirements.txt   # Python dependencies
|-- README.md          # Project documentation

(Update this to match your actual files.)

Installation
Clone the repository
bash
   git clone https://github.com/srijamaradugula-byte/cropyeild-forecast.git
   cd cropyeild-forecast
(Optional) Create a virtual environment
bash
   python -m venv venv
   venv\Scripts\activate
Install the dependencies
bash
   pip install -r requirements.txt
Usage

Run the main script:

bash
python code.py

Follow the prompts, or edit the input values in the script, to get a predicted yield.

Results
Model	R2 Score	MAE	RMSE
(Model 1)	-	-	-
(Model 2)	-	-	-

(Fill in your results and add screenshots of your charts or output here.)

Future Scope
Add live weather and soil data through APIs
Build a web interface for farmers using Flask or Streamlit
Try advanced models such as XGBoost or deep learning
Add support for more crops and regions
Author

Maradugula Srija GitHub: @srijamaradugula-byte
