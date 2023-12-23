#Dec22_2023
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import f1_score
import sys
import time
import math
import torch   
import pandas_datareader as web
import pandas  as pd
import pymysql
import tensorflow as tf
import scipy
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras import Sequential
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from numpy.random import seed
from nptdms import TdmsFile
#Start Coding Here#
from keras.initializers import HeUniform

tf.random.set_seed(64)
np.random.seed(64)

def func_f1_Score(title, actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    f1_LSTM = []
    matrix = confusion_matrix(actual, predicted, labels=[1, 0])
    print(title, ': \n', matrix)
    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1, 0]).reshape(-1)
    print(title,  ':\n', tp, fn, fp, tn)
    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(actual, predicted, labels=[1, 0])
    print(title, ': \n', matrix)
    f1_LSTM = f1_score(actual, predicted,average='weighted')

    format_f1_LSTM = "{:.5f}".format(f1_LSTM)
    print('===========================>=', format_f1_LSTM)
    return format_f1_LSTM

if __name__ == '__main__':

    path  = '/data/jkim/Rail_road/MxV_Main_Files/TCI_Test/'
    file = 'test08.10.10khz.od4p15__UTC_20220810_211247.641.tdms'
    tdms_file = TdmsFile( path+ file)
       #"C:/MxData/test08.tdms")
    # Extract data from channels
    data = {}
    for channel in tdms_file.groups()[0].channels():
        data[channel.name] = channel.data
# Convert to Pandas DataFrame
    df_5000 = pd.DataFrame(data)
    print (df_5000)
   # print('df', df.iloc[:5000])
    #df_5000 = df.iloc[:5000]
    np_df = np.array(df_5000)
    #print ('(1) Shape of Data' ,   np_df.shape)
    #print ('(2) Contents of data', np_df )
    Tr_D = []
    Tr_L = []
    TC_Test_D  = []
    TC_Test_L  = []
    NC_Test_D  = []
    NC_Test_L  = []

    #----Train Position Data Collection---------#
    def Data_parapration( s_Ch ,e_Ch, Tr_Dx, Tr_Lx, Lbx):
        for n in range (s_Ch ,e_Ch ):
            np_dataset = np_df[:,[n]]
            np_aSingle_Shape =  np_dataset.reshape(-1) #Reshape to the single Deminsion
            Tr_Dx.append(np_aSingle_Shape.tolist()) #Conterver it to the list
            Tr_Lx.append(Lbx)
        return Tr_Dx, Tr_Lx
    #--------------------------------------------------#
    #--------------------------------------------------#
    #  700 - 800 ; Normal Condition
    #  3000 - 3100 ; Training Position
    #------( Training Data )---------------------------------#
    s_Ch = 700  ; e_Ch=750 ; Lb =0
    Tr_D, Tr_L = Data_parapration(s_Ch, e_Ch, Tr_D, Tr_L, Lb)

    s_Ch = 3000  ; e_Ch=3050 ; Lb =1
    Tr_D, Tr_L = Data_parapration(s_Ch, e_Ch, Tr_D, Tr_L, Lb)

    #-------( Test Data )-----------------------------------#
    s_Ch = 750  ; e_Ch=800 ; Lb =0
    NC_Test_D, NC_Test_L = Data_parapration(s_Ch, e_Ch, NC_Test_D, NC_Test_L, Lb)

    s_Ch = 3050  ; e_Ch=3100 ; Lb =1
    TC_Test_D, TC_Test_L = Data_parapration(s_Ch, e_Ch, TC_Test_D, TC_Test_L, Lb)

#---Training----------------#
    X = np.array(Tr_D)
    y = np.array(Tr_L)  # Modify the labels to binary

    from keras.callbacks import ReduceLROnPlateau

    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Reshape the input data for the LSTM
    #from keras.optimizers import AdaBound
    from keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(GRU(65, return_sequences=True, input_shape=(1, X.shape[2])))
    model.add(GRU(40, return_sequences=False,kernel_initializer=HeUniform()))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='Adamax', metrics=['accuracy'] )
    model.fit(X, y, epochs=1000 , batch_size=1000, verbose=2,callbacks=[reduce_lr])

    #model.compile(loss='sparse_categorical_crossentropy',optimizer=AdaBound(lr=0.001, final_lr=0.0001), metrics=['accuracy'] )

    #model.fit(X, y, epochs=1000 , batch_size=10000, verbose=2,

    #---Test Train Position---#
    TX_test = np.array(TC_Test_D)
    Ty_test = np.array(TC_Test_L)
    TX_test = np.reshape(TX_test, (TX_test.shape[0], 1, TX_test.shape[1]))

    Ty_pred = model.predict(TX_test)
    Ty_pred_binary = np.argmax(Ty_pred, axis=1)
    print(Ty_pred_binary, Ty_test)
    finalResult = func_f1_Score('<GRU> Train Position',  Ty_pred_binary,TC_Test_L)
    print(finalResult)

    # ---Normal Condition ---#
    NX_test = np.array(NC_Test_D)
    Ny_test = np.array(NC_Test_L)
    NX_test = np.reshape(NX_test, (NX_test.shape[0], 1, NX_test.shape[1]))
    # Make predictions on the test data
    Ny_pred = model.predict(NX_test)
    Ny_pred_binary = np.argmax(Ny_pred, axis=1)
    print(Ny_pred_binary, Ny_test)
    finalResult = func_f1_Score('<GRU> Normal Position', Ny_pred_binary,NC_Test_L)
    print(finalResult)
#-------------------------------------------------------------------#
