from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
import pickle
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from keras.optimizers import Adam

main = tkinter.Tk()
main.title("Detection & Prediction of Comorbidities of Diabetes using Machine Learning Techniques") 
main.geometry("1300x1200")

global dataset, X, Y, X_train, y_train, X_test, y_test, ann_model
global accuracy, precision, recall, fscore, labels
global scaler

def loadData():
    global dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))
    labels = ['Normal', 'Comorbidity Detected']
    label = dataset.groupby('Outcome').size()
    label.plot(kind="bar")
    plt.xlabel("Disease Type 0 (Normal) & 1 (Comorbidity)")
    plt.ylabel("Count")
    plt.title("Normal, Heart & Diabetes (Comorbidity) Disease Graph")
    plt.show()
    
def datasetProcessing():
    text.delete('1.0', END)
    global dataset, label_encoder, scaler, X, Y, X_train, y_train, X_test, y_test
    dataset.fillna(0, inplace = True)

    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset after preprocessing & normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runNaiveBayes():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, cnn_model
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predict = nb.predict(X_test)
    calculateMetrics("Naive Bayes", y_test, predict)

def runANN():
    global accuracy, precision, recall, fscore, ann_model
    global X_train, y_train, X_test, y_test
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)

    ann_model = Sequential()
    ann_model.add(Dense(200, input_shape=(X_train.shape[1],), activation='relu', name='fc1'))
    ann_model.add(Dense(200, activation='relu', name='fc2'))
    ann_model.add(Dense(y_train1.shape[1], activation='softmax', name='output'))
    optimizer = Adam(learning_rate=0.001)
    ann_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #train and load the model
    if os.path.exists("model/ann_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
        hist = ann_model.fit(X_train, y_train1, batch_size = 8, epochs = 50, validation_data=(X_test, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/ann_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        ann_model.load_weights("model/ann_weights.hdf5")
    predict = ann_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    calculateMetrics("ANN", y_test, predict)
    
def graph():
    df = pd.DataFrame([['Naive Bayes','Accuracy',accuracy[0]],['Naive Bayes','Precision',precision[0]],['Naive Bayes','Recall',recall[0]],['Naive Bayes','FSCORE',fscore[0]],
                       ['ANN','Accuracy',accuracy[1]],['ANN','Precision',precision[1]],['ANN','Recall',recall[1]],['ANN','FSCORE',fscore[1]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()

def predictDisease():
    text.delete('1.0', END)
    global scaler, ann_model, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    temp = dataset.values
    dataset = dataset.values
    X = scaler.transform(dataset)
    predict = ann_model.predict(X)
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        text.insert(END,"Test Data = "+str(temp[i])+" =====> Predicted As "+str(labels[pred])+"\n\n")
    

font = ('times', 16, 'bold')
title = Label(main, text='Detection & Prediction of Comorbidities of Diabetes using Machine Learning Techniques')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Comorbidity Diabetes Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=datasetProcessing)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
nbButton.place(x=330,y=150)
nbButton.config(font=font1) 

annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=630,y=150)
annButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Disease from Test Data", command=predictDisease)
predictButton.place(x=330,y=200)
predictButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
