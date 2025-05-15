import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.models import model_from_json
df=pd.read_csv("pima-indians-diabetes.csv",header=None)
print(df)
x=df.iloc[:,0:8].values
y=df.iloc[:,8].values
model=Sequential()
layers12=Dense(10,input_shape=(8,),activation="relu")
layers3=Dense(20,activation="relu")
layers4=Dense(1,activation="sigmoid")
model.add(layers12)
model.add(layers3)
model.add(layers4)
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x,y,epochs=100,batch_size=10)
model_json=model.to_json()
with open("rajesh1.model_json","w")as f1:
    f1.write("model_json")
model.save_weights("model.weights.h5")