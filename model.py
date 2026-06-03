#Logging
import logging
import mlflow
import mlflow.sklearn
import warnings
#Data manipulation
import pandas as pd
import numpy as np
#Modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder , OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
#Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('mlflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)



#Load Data
df = pd.read_csv("data/taxi_trip_pricing.csv")


#Drop missing target
df = df.dropna(subset=['Trip_Price'])



#Split featurs and target
X, y = df.drop(columns=['Trip_Price']), df['Trip_Price']

#Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Feature Engineer Columns
class AddFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X['speed_kmph'] = X['Trip_Distance_km'] / X['Trip_Duration_Minutes'] * 60
        return X

# Define columns
numeric_cols     = ['Trip_Distance_km', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes','speed_kmph']
discrete_cols    = ['Passenger_Count']
ordinal_cols     = ['Traffic_Conditions']
nominal_cols     = ['Time_of_Day', 'Day_of_Week', 'Weather']


# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps =[('imputer', SimpleImputer(strategy ='mean')),('scaler', RobustScaler())]) , numeric_cols),
        ('dis', Pipeline(steps=[('imputer', SimpleImputer(strategy ='most_frequent')), ('scaler', RobustScaler())]), discrete_cols),
        ('ord', Pipeline(steps=[('imputer', SimpleImputer(strategy ='most_frequent')), ('encoder', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]))]), ordinal_cols),
        ('nom', Pipeline(steps=[('imputer', SimpleImputer(strategy ='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]), nominal_cols)
    ]
)

models = {'LinearRegression':LinearRegression() , 
          'Ridge':Ridge() ,
          'Lasso':Lasso() , 
          'RandomForest':RandomForestRegressor(), 
          'XGBoost':XGBRegressor(),
          'NeuralNetwork':MLPRegressor(solver='lbfgs',max_iter=5000)
}

mlflow.set_experiment("taxi-fare-regression")


for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline= Pipeline(steps=[  
        ('feature_engineer',AddFeature()),
        ('preprocessor',preprocessor),
        ('model',model)
        ])
        mlflow.log_param('model',name)
        cv_mae  = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        cv_rmse = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
        cv_r2   =  cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2').mean()
        pipeline.fit(X_train, y_train)

        #Feature Selection Columns --> If needed (Not for this project)

        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pipeline.named_steps['model'].feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            top5 = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"{name} top 5 features: {top5}")
            mlflow.log_metrics({f"imp_{k}": v for k, v in importance_dict.items()})


        y_pred_train = pipeline.predict(X_train)
        train_mae  = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2   = r2_score(y_train, y_pred_train)


        y_pred = pipeline.predict(X_test)
        test_mae  = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2   = r2_score(y_test, y_pred)

        mlflow.log_metrics({
            'cv_mae': cv_mae,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        })


        mlflow.sklearn.log_model(pipeline, name='model')

        logger.info(
            f"{name} | "
            f"CV → MAE: {cv_mae:.2f} RMSE: {cv_rmse:.2f} R²: {cv_r2:.4f} | "
            f"Train → MAE: {train_mae:.2f} RMSE: {train_rmse:.2f} R²: {train_r2:.4f} | "
            f"Test → MAE: {test_mae:.2f} RMSE: {test_rmse:.2f} R²: {test_r2:.4f} | "
            f"Overfit gap R²: {train_r2 - test_r2:.4f}"
        )

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0])
    }
    
    pipeline= Pipeline(steps=[  
        ('feature_engineer',AddFeature()),
        ('preprocessor',preprocessor),
        ('model',RandomForestRegressor(**params,random_state=42))
        ])
    cv_mae  = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    return cv_mae
    

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)
logger.info(f"Best params: {study.best_params}")



with mlflow.start_run(run_name='RandomForest-Tuned'):
    pipeline= Pipeline(steps=[  
        ('feature_engineer',AddFeature()),
        ('preprocessor',preprocessor),
        ('model',RandomForestRegressor(**study.best_params,random_state=42))
        ])
    mlflow.log_params(study.best_params)
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    train_mae  = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2   = r2_score(y_train, y_pred_train)


    y_pred_test = pipeline.predict(X_test)
    test_mae  = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2   = r2_score(y_test, y_pred_test)

    mlflow.log_metrics({
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    })
    
    mlflow.sklearn.log_model(pipeline, name='model')

    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "taxi-fare-best")
    logger.info(f"Tuned RandomForest → Test R²: {test_r2:.4f} | Test MAE: {test_mae:.2f}")




