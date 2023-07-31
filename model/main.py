import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    xtrain, xtest, ytrain, ytest = tts(X,y, train_size=0.8, random_state=42)
    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    print(f'Accuracy of our model : {accuracy_score(ytest, y_pred)}')
    print(f'Classification Report :\n{classification_report(ytest, y_pred)}')
    return model, scaler

def get_clean_data(data):
    data = data.drop(columns=['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B' : 0})
    return data

def main():
    data = pd.read_csv("data/data.csv")
    data = get_clean_data(data)
    model, scaler = create_model(data)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
