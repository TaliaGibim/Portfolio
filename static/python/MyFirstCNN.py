# %%
import numpy as np
import pandas as pd
import pickle
import os

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
        for dim in dimen:
            w_b = {'w':np.random.random(dim)*scaler,'b':np.zeros((dim[0],1))}
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
            layerI = len(dimen)-i-1
            if i !=0:
                dz = np.multiply(da,d_relu(hiden['z']))
            da = np.dot(layer['w'].T,dz)
            dw = 1/y.shape[1] * (np.dot(dz,hiden['a'].T))
            db = 1/y.shape[1] * (np.sum(dz,axis = 1,keepdims = True))
            self.layers[layerI]['w'] = layer['w'] - self.alpha *dw
            self.layers[layerI]['b'] = layer['b'] - self.alpha *db 
            
    def lossfunction(self):
        lines = np.arange(y.shape[1])                     #Array o to number of observations
        columns = np.argmax(y, axis=0, keepdims = True)     #Index of the true label
        self.softmax = self.softmax.T
        crossentropy = self.softmax[ lines , columns ]
        self.loss = -np.sum(np.log(crossentropy))
        
        #Acuracy
        predictions = np.argmax(self.softmax.T, axis=0, keepdims = True)
        self.acuracy = np.sum(predictions == columns )/ x.shape[1] 
        
    def learning(self,interactions,x,y):
        for i in range(0,interactions):
            self.foward(x)
            self.backpropagation(y)
            self.lossfunction()
            print(self.loss,str(i)+"  ",self.acuracy)
            
    def predict(self,image):
            self.foward(image)
            print(self.softmax)
            self.prediction = np.argmax(self.softmax)
            self.accuracy = self.softmax[self.prediction]
           
# %% New object

if __name__ == "__main__":
     
    # Loading the database
    train_df = pd.read_csv(os.path.join('static','database','train.csv'), header=None)
    test_df = pd.read_csv(os.path.join('static','database','test.csv'), header=None)

    # %% Defining parameters
    #dimen = [(128,784),(64,128),(32,64),(10,32)]
    dimen = [(10,784)]

    # %%
    teste = train_df.iloc[:,1:785]
    x = np.array((teste.T)/255)

    image =  np.array((train_df.iloc[0:1,1:785].T)/255)

    # %% Doing Vector h_hat

    count = 0
    y_database = train_df.iloc[0:60000, 0:1].T.values  # Transpose and get values as a NumPy array
    y = np.zeros((10, 60000))

    for i in y_database[0]:  # Use y_database[0] to access the single row of labels
        y[i, count] = 1
        count += 1
        
    model = MLP(dimen,0.5,0.01)
    model.learning(5, x, y)
    model.predict(image)
    pickle.dump(model, open(os.path.join('static','model','model.pkl'),'wb'))



