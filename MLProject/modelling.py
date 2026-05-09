import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import shutil
import os

def train():
    print("Memuat data preprocessing...")
    train_df = pd.read_csv('california_housing_preprocessing/train_processed.csv')
    
    X_train = train_df.drop('median_house_value', axis=1)
    y_train = train_df['median_house_value']

    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        print("Melatih model...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        print("Menyimpan artefak model untuk Docker...")
        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")
            
        mlflow.sklearn.save_model(model, "saved_model")
        print("Model berhasil disimpan di folder 'saved_model'.")

if __name__ == "__main__":
    train()