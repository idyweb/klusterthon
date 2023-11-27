import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("Crop_Data.csv")
df.head()

df.rename(columns={'harvest season': 'harvest_season'}, inplace=True)


label_encoder = LabelEncoder()
country_encoder = LabelEncoder()
season_encoder = LabelEncoder()

df['label'] = label_encoder.fit_transform(df['label'])
df['Country'] = country_encoder.fit_transform(df['Country'])
df['harvest_season'] = season_encoder.fit_transform(df['harvest_season'])



# Shuffle the DataFrame rows
shuffled_df = df.sample(frac=1, random_state=2023)


# Scale numerical columns (temperature, humidity, pH, water availability)
scaler = MinMaxScaler()
columns_to_scale = ['temperature', 'humidity', 'ph', 'water availability']
shuffled_df[columns_to_scale] = scaler.fit_transform(shuffled_df[columns_to_scale])



X = shuffled_df.drop(['harvest_season'], axis=1)
y = shuffled_df['harvest_season']

# Splitting the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=8)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_classifier.predict(X_test)

# Assess the model performance
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)



from joblib import dump

dump(knn_classifier, 'knn_model.pkl')
dump(label_encoder, 'label_encoder.pkl')
dump(country_encoder, 'country_encoder.pkl')
dump(season_encoder, 'season_encoder.pkl')
dump(scaler, 'minmax_scaler.pkl')
