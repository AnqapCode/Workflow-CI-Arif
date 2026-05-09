import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

def train():
    print("Memuat data preprocessing...")
    train_df = pd.read_csv('california_housing_preprocessing/train_processed.csv')
    
    X_train = train_df.drop('median_house_value', axis=1)
    y_train = train_df['median_house_value']

    # Autolog akan otomatis menyimpan model ke folder 'mlruns/0/<RUN_ID>/artifacts/model'
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        print("Melatih model...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        print("Selesai! Model otomatis tersimpan di dalam folder mlruns.")

if __name__ == "__main__":
    train()