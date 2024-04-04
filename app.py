import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")
app.debug = True

df_1 = pd.read_csv("first_telc.csv")

# Define default values for form inputs
DEFAULT_VALUES = {
    'query1': '0',  # SeniorCitizen
    'query2': '0.0',  # MonthlyCharges
    'query3': '0.0',  # TotalCharges
    'query4': '',  # gender
    'query5': '',  # Partner
    'query6': '',  # Dependents
    'query7': '',  # PhoneService
    'query8': '',  # MultipleLines
    'query9': '',  # InternetService
    'query10': '',  # OnlineSecurity
    'query11': '',  # OnlineBackup
    'query12': '',  # DeviceProtection
    'query13': '',  # TechSupport
    'query14': '',  # StreamingTV
    'query15': '',  # StreamingMovies
    'query16': '',  # Contract
    'query17': '',  # PaperlessBilling
    'query18': '',  # PaymentMethod
    'query19': '0'  # tenure
}

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Get form data and handle missing keys with default values
        inputQuery1 = request.form.get('query1', DEFAULT_VALUES['query1'])
        inputQuery2 = request.form.get('query2', DEFAULT_VALUES['query2'])
        inputQuery3 = request.form.get('query3', DEFAULT_VALUES['query3'])
        inputQuery4 = request.form.get('query4', DEFAULT_VALUES['query4'])
        inputQuery5 = request.form.get('query5', DEFAULT_VALUES['query5'])
        inputQuery6 = request.form.get('query6', DEFAULT_VALUES['query6'])
        inputQuery7 = request.form.get('query7', DEFAULT_VALUES['query7'])
        inputQuery8 = request.form.get('query8', DEFAULT_VALUES['query8'])
        inputQuery9 = request.form.get('query9', DEFAULT_VALUES['query9'])
        inputQuery10 = request.form.get('query10', DEFAULT_VALUES['query10'])
        inputQuery11 = request.form.get('query11', DEFAULT_VALUES['query11'])
        inputQuery12 = request.form.get('query12', DEFAULT_VALUES['query12'])
        inputQuery13 = request.form.get('query13', DEFAULT_VALUES['query13'])
        inputQuery14 = request.form.get('query14', DEFAULT_VALUES['query14'])
        inputQuery15 = request.form.get('query15', DEFAULT_VALUES['query15'])
        inputQuery16 = request.form.get('query16', DEFAULT_VALUES['query16'])
        inputQuery17 = request.form.get('query17', DEFAULT_VALUES['query17'])
        inputQuery18 = request.form.get('query18', DEFAULT_VALUES['query18'])
        inputQuery19 = request.form.get('query19', DEFAULT_VALUES['query19'])
    except KeyError as e:
        # Handle missing keys
        # Log the error or provide a default response
        # Example: return an error message
        return "An error occurred: {}".format(e)

    # Load the model
    model = pickle.load(open("model.sav", "rb"))

    # Create DataFrame with form data
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Ensure data types match
    new_df = new_df.astype({'SeniorCitizen': int, 'MonthlyCharges': float, 'TotalCharges': float})

    # Continue with the rest of the code...
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    # Drop columns customerID and tenure
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # Perform one-hot encoding
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Make prediction
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probability * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probability * 100)

    return render_template('home.html', output1=o1, output2=o2,
                           query1=inputQuery1,
                           query2=inputQuery2,
                           query3=inputQuery3,
                           query4=inputQuery4,
                           query5=inputQuery5,
                           query6=inputQuery6,
                           query7=inputQuery7,
                           query8=inputQuery8,
                           query9=inputQuery9,
                           query10=inputQuery10,
                           query11=inputQuery11,
                           query12=inputQuery12,
                           query13=inputQuery13,
                           query14=inputQuery14,
                           query15=inputQuery15,
                           query16=inputQuery16,
                           query17=inputQuery17,
                           query18=inputQuery18,
                           query19=inputQuery19)


   

app.run()
