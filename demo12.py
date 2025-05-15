import pandas as pd
import numpy as np
from keras.models import model_from_json
df=pd.read_csv("pima-indians-diabetes.csv",header=None)
print(df)
x=df.iloc[:,0:8].values
y=df.iloc[:,8].values
json_file= open("rajesh1.model_json","r")
new_model=json_file.read()
model=model_from_json(new_model)
model.load_weights("model.weights.h5")
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
final=model.evaluate(x,y)
print(final)