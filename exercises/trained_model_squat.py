import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn import preprocessing


df = pd.read_csv('squat_dataset.csv')

#print(df.head(-10))

#print(df['Label'].value_counts())

label_encoder = preprocessing.LabelEncoder()

obj = (df.dtypes == 'object')
for col in list(obj[obj].index):
    df[col] = label_encoder.fit_transform(df[col])

#print(df)

X = df.drop(['Label'],axis=1)
Y = df['Label']

X_train,X_test,y_train,y_test = train_test_split(X, Y ,
                                                  test_size=0.2, random_state= 42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

#print(X)


rf_model = RandomForestClassifier()
rf_model.max_iter= 10000
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#print(accuracy_score(y_test,y_pred)*100,'%')


with open('squat_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)