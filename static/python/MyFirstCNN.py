# %%
import numpy as np
import pandas as pd
import pickle
import os
import itertools
import bz2file as bz2

# %% Setinng Relu function and Sigmoid Function
def relu(x):
	return np.maximum(x, 0)

def sig(x):
    return 1/(1+ np.exp(-x))

# %% Setinng Derivation of Relu function and Sigmoid Function

def d_relu(x):
	return np.maximum(1, 0)

# %% Defining class
              
class MLP:
    def __init__(self, dimen, alpha, scaler):
        self.layers = []
        self.alpha = alpha
        self.prediction = None  
        self.accuracy = None
        self.nLayer = len(dimen)-1
        for dim in dimen:
            a , b = dim
            #w_b = {'w':np.random.random(dim)*scaler,'b':np.zeros((dim[0],1))}
            w_b = {'w':np.random.rand(a,b)*scaler,'b':np.zeros((a,1))}
            self.layers.append(w_b)
        
    def foward(self,x):
        self.hiddenstate = []
        self.softmax = []
        a_z = {}
        for i,layer in enumerate(self.layers):
            a_z = {'a':x, 'z': np.dot(layer["w"],x) + layer["b"] }
            self.hiddenstate.append(a_z)
            x = relu(self.hiddenstate[i]['z'])
        
        # np.exp(x)/np.sum(np.exp(y))
        final_Label = self.hiddenstate[-1]['z']
        a_final = np.exp(final_Label) / np.sum(np.exp(final_Label), axis = 0, keepdims = True)
        
        self.softmax.append(a_final)
        self.softmax = np.squeeze(np.array(self.softmax), axis=0)

    def backpropagation(self,y):
        
        dz =  self.softmax - y
        for i, (hiden,layer) in enumerate(zip(reversed(self.hiddenstate),reversed(self.layers))):
            layerI = self.nLayer - i
            if i !=0:
                dz = np.multiply(da,d_relu(hiden['z']))
            da = np.dot(layer['w'].T,dz)
            dw = 1/y.shape[1] * (np.dot(dz,hiden['a'].T))
            db = 1/y.shape[1] * (np.sum(dz,axis = 1,keepdims = True))
            self.layers[layerI]['w'] = layer['w'] - self.alpha *dw
            self.layers[layerI]['b'] = layer['b'] - self.alpha *db 
            
    def lossfunction(self,x,y):
        lines = np.arange(y.shape[1])                     #Array o to number of observations
        columns = np.argmax(y, axis=0, keepdims = True)     #Index of the true label
        self.softmax = self.softmax.T
        crossentropy = self.softmax[ lines , columns ]
        self.loss = -np.sum(np.log(crossentropy))
        
        #Accuracy
        predictions = np.argmax(self.softmax.T, axis=0, keepdims = True)
        self.accuracy = np.sum(predictions == columns )/ x.shape[1] 
        
    def modelAccuracy(self,x_test,y_target):
        self.foward(x_test)
        predi = np.argmax(self.softmax, axis=0, keepdims = True)
        self.modelAccuracy = np.sum(predi == y_target )/ x_test.shape[1] 
        
    def learning(self,interactions,x,y):
        for i in range(0,interactions):
            self.foward(x)
            self.backpropagation(y)
            self.lossfunction(x,y)
            #print(self.loss,str(i)+"  ",self.accuracy)
            
    def predict(self,image):
            self.foward(image)
            print(self.softmax)
            self.prediction = np.argmax(self.softmax)
            self.accuracy = self.softmax[self.prediction]

if __name__ == "__main__":
     
    # Loading the database
    train_df = pd.read_csv(os.path.join('static','database','train.csv'), header=None)
    test_df = pd.read_csv(os.path.join('static','database','test.csv'), header=None)

    # %%
    train = train_df.iloc[:,1:785]
    x_train = np.array((train.T)/255)

    test = test_df.iloc[:,1:785]
    x_test = np.array((test.T)/255)

    image =  np.array((train_df.iloc[0:1,1:785].T)/255)

    # %% Doing Vector h_hat

    count = 0
    y_database = train_df.iloc[0:60000, 0:1].T.values  # Transpose and get values as a NumPy array
    y_train = np.zeros((10, 60000))

    for i in y_database[0]:  # Use y_database[0] to access the single row of labels
        y_train[i, count] = 1
        count += 1
        
    y_test = test_df.iloc[0:60000, 0:1].T.values

    # %% Defining parameters

    scaler = [0.005 , 0.01]
    alpha =	[0.125 , 0.5 , 1 ]
    interactions = [20 , 40 , 80]
    architectury = [ [(10,784)] , [(100,784),(10,100)] , [(128,784),(64,128),(10,64)]]

    # Get all combinations
    combinations = list(itertools.product(scaler, alpha, interactions ,architectury))

    # Convert to a DataFrame
    df = pd.DataFrame(combinations, columns=['scaler', 'alpha','interactions','architectury'])
    df["AccuracyTrain"] = 0
    df["AccuracyTest"] = 0

    # %% Grid Search

    for index, row in df.iterrows():
        print(index)
        model = MLP(row['architectury'],row['alpha'],row['scaler'])
        model.learning(row['interactions'], x_train, y_train)
        df.at[index, 'AccuracyTrain'] = model.accuracy
        model.modelAccuracy(x_test,y_test)
        df.at[index, 'AccuracyTest'] = model.modelAccuracy
        
    pickle.dump(df, open('./result.pkl','wb'))
    # %% Exporting the best model

    model = MLP([(10,784)],1,0.005)
    model.learning(80, x_train, y_train)
    with bz2.BZ2File("model.pkl.bz2", "wb") as f:
        pickle.dump(model, f)
    




