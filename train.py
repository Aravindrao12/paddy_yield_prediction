import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_percentage_error
import joblib
import os
from sklearn.model_selection import cross_val_score

datapath=r"C:\Users\Infinix\Desktop\python.py\paddy.py\paddydataset.csv"
target="Paddy yield(in Kg)"
drop_col=[target,"Trash(in bundles)"]
corr_thersold=0.95

df=pd.read_csv(datapath)
df.columns=df.columns.str.strip()
df=df.drop_duplicates()

x=df.drop(columns=drop_col)
y=df[target]


num_all = x.select_dtypes(include="number")
corr=num_all.corr().abs()
upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
drop_col=[c for c in upper.columns if (upper[c]>corr_thersold).any()]

x_reduced=x.drop(columns=drop_col)
cat_cols= x_reduced.select_dtypes(exclude="number").columns.tolist()
num_cols=x_reduced.select_dtypes(include="number").columns.tolist()

x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.20,random_state=42)


preprocessor=ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_cols),
        ("num","passthrough",num_cols),
    ]
)
pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestRegressor(random_state=42,n_jobs=-1)),
])
pipeline.fit(x_train,y_train)

train_pred=pipeline.predict(x_train)
y_pred=pipeline.predict(x_test)


print(f"Train r2 :{r2_score(y_train,train_pred):.4f}")
print(f"Test r2:{r2_score(y_test,y_pred):.4f}")
print(f"rsme:{root_mean_squared_error(y_test,y_pred):.4f}")


cv_score=cross_val_score(pipeline,x_reduced,y,cv=5,scoring="r2",n_jobs=-1)
print(f"5-flod CV r2:{cv_score.mean():.4f} (+/-{cv_score.std():.4f})")

model_dir = r"C:\Users\Infinix\Desktop\python.py\paddy.py\models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(pipeline, os.path.join(model_dir, "paddy_pipeline.pkl"))
joblib.dump(x_reduced.columns.tolist(), os.path.join(model_dir, "input_columns.pkl"))
print("saved in:", model_dir)


print(x_reduced.columns.tolist())




# os.makedirs("models",exist_ok=True)
# joblib.dump(pipeline,"model/paddy_pipeline.pkl")
# joblib.dump(x_reduced.columns.tolist(),"model/input_columns.pkl")
# print("saved:model/paddy_pipeline.pkl and model/input_columns.pkl")
