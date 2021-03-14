import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#3.7

df = pd.read_csv("/Users/tahmidalam/Documents/University/Case Studies & AI/Assignment/Stars.csv")
#dataset found on: https://www.kaggle.com/deepu1109/star-dataset


print(df.head(5))

print(df.shape)
# 240 rows, 7 columns

print(df.columns)
# columns are: 'Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
#                          'Absolute magnitude(Mv)', 'Star type', 'Star color', 'Spectral Class'

# find the structure of the variables; are they continuous or categorical?
print(df.dtypes)
# Temperature & Star type are integers
#           star type can be thought to be categorical
# Luminosity, Radius and Absolute magnitude are floats (continuous?)
# star colour and Spectral Class are objects
#           both can be thought to be categorical


# now check for NA/missing values
print(df.isnull())

print(df['Star type'].unique())             # unique values: [0 1 2 3 4 5]              # this will be what we predict

# 0 = Brown dwarf       1 = Red dwarf       2 = white dwarf     3 = main sequence       4 = Supergiant          5 = Hypergiant



#           # Red' 'Blue White' 'White' 'Yellowish White' 'Blue white'
#                                               'Pale yellow orange' 'Blue' 'Blue-white' 'Whitish' 'yellow-white'
#                                                   'Orange' 'White-Yellow' 'white' 'Blue ' 'yellowish' 'Yellowish'
#                                                       'Orange-Red' 'Blue white ' 'Blue-White']

print(df['Spectral Class'].unique())        # [O B A F G K M]       this is the order for the harvard spectral classification, in order

# replace the spectral strings to numeric:

df['Spectral Class'] = df['Spectral Class'].map({'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M':6 })


print(df['Spectral Class'].unique)

# Change blue variations into 'Blue-White : Blue White, Blue white, Blue-white

df['Star color'] = df['Star color'].replace('Blue White', 'Blue-White')
df['Star color'] = df['Star color'].replace('Blue white', 'Blue-White')
df['Star color'] = df['Star color'].replace('Blue-white', 'Blue-White')
df['Star color'] = df['Star color'].replace('Blue white ', 'Blue-White')

df['Star color'] = df['Star color'].replace('Blue ', 'Blue')

df['Star color'] = df['Star color'].replace('white', 'White')

df['Star color'] = df['Star color'].replace('yellow-white', 'Yellow-White')

df['Star color'] = df['Star color'].replace('yellowish', 'Yellowish')

df['Star color'] = df['Star color'].replace('Yellow-White', 'White-Yellow')



#------------------------------------------
# perform data reduction / FAMD on R

#use heatmap

import seaborn as sns


#sns.heatmap(data = df.corr(), annot= True)

#plt.show()

#--------------------------------------------------
# convert string data to numeral data



#the colours are:   Blue-White, Blue, White, Yellow-White, Yellowish, White-Yellow

print(df.columns)

data = df

star_colour = pd.get_dummies(data["Star color"])
print(star_colour.columns)


data["Blue"] = star_colour["Blue"].to_list()
data["Blue_White"] = star_colour["Blue-White"].to_list()
data["Orange"] = star_colour["Orange"].to_list()
data["Orange_Red"] = star_colour["Orange-Red"].to_list()
data["Pale_Yellow_Orange"] = star_colour["Pale yellow orange"].to_list()
data["Red"] = star_colour["Red"].to_list()
data["White"] = star_colour["White"].to_list()
data["White_Yellow"] = star_colour["White-Yellow"].to_list()
data["Whitish"] = star_colour["Whitish"].to_list()
data["Yellowish"] = star_colour["Yellowish"].to_list()
data["Yellowish_White"] = star_colour["Yellowish White"].to_list()

del data['Star color']



print(data.columns)



#--------------------
# normalise the data:
#           Temperature (K), Luminosity(L/Lo), Radius(R/Ro)

from sklearn.preprocessing import StandardScaler

data.loc[:, df.columns != 'Star type'] = StandardScaler().fit_transform(data.loc[:, df.columns != 'Star type'])


print("HERE:")
print(data.head(5))

#all data but Target (spectral type) is normalised

#-------------------
#split data into test and train

x_data = data.loc[:, data.columns != 'Star type']
x_data = x_data.loc[:, x_data.columns != 'Star color']
y_data = data['Star type']


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.25, shuffle= True)

#print("xtrain: ", x_train.shape, " x_test: ", x_test.shape, " y train: ", y_train.shape, " y test: ", y_test.shape )

#----------------------------------------------------------------------------------
#build the neural net

#convert existing df's to numpy arrays for the modelling


print(x_train.columns)



x_train = np.column_stack((x_train['Temperature (K)'].values, x_train['Luminosity(L/Lo)'].values, x_train['Radius(R/Ro)'].values, x_train['Absolute magnitude(Mv)'].values,
                           x_train['Spectral Class'].values, x_train['Blue'].values, x_train['Blue_White'].values, x_train['Orange'].values, x_train['Orange_Red'].values,
                           x_train['Pale_Yellow_Orange'].values, x_train['Red'].values, x_train['White'].values, x_train['White_Yellow'].values, x_train['Whitish'].values,
                           x_train['Yellowish'].values, x_train['Yellowish_White'].values))

x_test = np.column_stack((x_test['Temperature (K)'].values, x_test['Luminosity(L/Lo)'].values, x_test['Radius(R/Ro)'].values, x_test['Absolute magnitude(Mv)'].values,
                           x_test['Spectral Class'].values, x_test['Blue'].values, x_test['Blue_White'].values, x_test['Orange'].values, x_test['Orange_Red'].values,
                           x_test['Pale_Yellow_Orange'].values, x_test['Red'].values, x_test['White'].values, x_test['White_Yellow'].values, x_test['Whitish'].values,
                           x_test['Yellowish'].values, x_test['Yellowish_White'].values))

y_train = np.array(y_train).astype(np.int)

y_test = np.array(y_test).astype(np.int)



#also shuffle the data to make sure there's no unintentional congestion:



#22 input variables, excluding the target variable.

model = keras.Sequential([
    keras.layers.Dense(16, input_shape= (16,), activation= 'relu'),
    keras.layers.Dense(16, activation= 'relu'),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dropout(rate= 0.25),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(16, activation= 'relu'),
    keras.layers.Dense(6, activation= 'softmax')
])

optimizer = keras.optimizers.Adam(learning_rate= 0.001, decay= 1e-6)

model.compile(optimizer= optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size= 4, epochs= 100, validation_data=(x_test, y_test))

print("Evaluation:")
model.evaluate(x_test, y_test, batch_size= 2)



plt.plot(history.history['accuracy'])
plt.legend(['accuracy'])
#plt.show()





