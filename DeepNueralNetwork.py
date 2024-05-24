import numpy as np
from math import pi,cos
import pickle as pkl
"""
    This is my assignment file for deep learning course,
besides full mathematical deduction I optimized numerical
stability by using Xavier initiate and Cosine annealing.
"""

class CosineAnnealing:
    def __init__(self,eta_min,eta_max,T):
        '''
        CosineAnnealing scheduler for updating learning rate
        :param eta_min: Minimum Learning Rate
        :param eta_max: Maximum Learning Rate
        :param T: Term of scheduling
        '''
        self.T = T
        self.eta_min = eta_min
        self.eta_max = eta_max
    def get_rate(self,cur):
        cur = cur%self.T + 1
        return self.eta_min + 0.5*(self.eta_max-self.eta_min)*(1+cos((cur/self.T)*pi))

class DNN:
    '''
        NOTICE:
        I compute gradients using sauare erro loss!!
    '''
    def __init__(self,layers:list,act='relu',alpha=None,reg=True):
        '''
        Inilization of Deep Neural Network
        :param layers: a list describes how many neurons in each layer
        :param act: Type of activation functions' type(relu, tanh, sigmoid are supported)
        :param lr: learning rate of parameters' updating
        :param alpha: If using leakyrelu activation,must be specified
        :param reg : IF True,activate function in the last layer will be erased.
        '''
        assert act in ('relu','sigmoid','tanh','leakyrelu'),f'Invalid activation type:"{act}",\nmust choose one of relu,sigmoid or tanh or leakyrelu'
        if act=='leakyrelu':
            if alpha == None:
                raise f'ERRO:leakyrelu\'s alpha didn\'t be specified!'
            else:
                self._alpha = alpha
        self.reg = reg
        self._layers = layers
        self._num_layers = len(layers) # Number of layers
        self._act = act
        self.weights,self.biases = self._init_weights_biases() # Xavier Initialization

    def _init_weights_biases(self):
        # There are num_layers-1 of weights matrixes and biases vector
        # Using dictionary is faster
        weights = {}
        biases = {}
        for i in range(self._num_layers-1):
            shape_weights,shape_biases = (self._layers[i+1],self._layers[i]),(self._layers[i+1])
            # compute xavier boundar
            limit = np.sqrt(6.0 / (self._layers[i+1] + self._layers[i]))
            # weights[i] = np.random.normal(0, limit, shape_weights)
            # biases[i] = np.random.normal(0, limit, shape_biases)
            weights[i] = np.random.uniform(0, limit, shape_weights)
            biases[i] = np.random.uniform(0, limit, shape_biases)
        return weights,biases

    def _f(self,u,diff):
        # combination of all kinds of activation functions
        if self._act == 'relu':
            return self._relu(u,diff)
        elif self._act == 'sigmoid':
            return self._sigmoid(u,diff)
        elif self._act == 'tanh':
            return self._tanh(u,diff)
        elif self._act == 'leakyrelu':
            return self._leakyrelu(u,diff)

    def _relu(self,u,diff=False):
        '''
        relu activation function
        :param u: input parameter
        :param diff: False if want to get function output
                    True if want to take derivative output
        :return: output of function output ar derivative output
        '''
        if not diff:
            return np.where(u < 0, 0, u)
        else:
            return np.where(u > 0, 1, 0)

    def _leakyrelu(self,u,diff=False):
        '''
        leakyrelu activation function
        :param u: input parameter
        :param diff: False if want to get function output
                    True if want to take derivative output
        :return: output of function output ar derivative output
        '''
        if not diff:
            return np.where(u < 0, self._alpha*u, u)
        else:
            return np.where(u > 0, 1, -self._alpha)

    def _sigmoid(self,u,diff=False):
        '''
        sigmoid activation function
        :param u: input parameter
        :param diff: False if want to get function output
                    True if want to take derivative output
        :return: output of function output ar derivative output
        '''
        f = 1 / (1 + np.exp(-u))
        if not diff:
            return f
        else:
            return f * (1-f)

    def _tanh(self, u, diff=False):
        '''
        tanh activation function
        :param u: input parameter
        :param diff: False if want to get function output
                    True if want to take derivative output
        :return: output of function output ar derivative output
        '''
        if not diff:
            return (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))
        else:
            return 4 * np.exp(2*u) / (np.exp(2*u) + 1) ** 2

    def _forward(self,x):
        if not isinstance(x,np.ndarray) or x.shape == ():
            x = np.array([x])
        # forward propagate
        A, Z = {} ,{} # storage outputs of each layers
        a = x.copy()
        # There is no activation in first layer
        A[0],Z[0] = a,a
        for idx,W in self.weights.items():
            z = W @ a + self.biases[idx]
            a = self._f(z,diff=False)
            A[idx+1],Z[idx+1] = a,z
        if self.reg:
            return z,A,Z
        else:
            return a,A,Z

    def _backward(self,x,y):
        if not isinstance(x,np.ndarray) or x.shape == ():
            x = np.array([x])
        if not isinstance(y,np.ndarray) or y.shape == ():
            y = np.array([y])
        # forward compute
        y_predict,A,Z = self._forward(x)
        # index of the last layer
        L = self._num_layers - 1
        # compute erro
        if not self.reg:
            erro = self._diag(Z[L]) @ (y_predict-y)
        else:
            erro = (y_predict - y)

        for l in range(L-1,-1,-1):
            self.biases[l] = self.biases[l] - self.lr * erro
            for j in range(self.weights[l].shape[1]):
                self.weights[l][:, j] = self.weights[l][:, j] - self.lr * A[l][j] * erro
            # update erro
            erro = self._diag(Z[l]) @ (self.weights[l].T) @ erro

    def _diag(self,x):
        if not isinstance(x,np.ndarray):
            x = np.array([x])
        # compute diagonse matrix
        if x.shape==():
            # if x is a scalar then transfer it to a vector
            x = np.array([x])
        return np.diag(self._f(x,diff=True))

    def _loss(self,y_pred,y_true):
        return (y_pred-y_true)@(y_pred-y_true) / 2

    def predict(self,x):
        # prediction
        return self._forward(x)[0]

    def fit(self,X,y,max_iter=1000,scheduler=None,early_stopping_patience=30):
        train_loss = []
        best_loss = np.inf
        counter = 0
        for epoch in range(1,max_iter+1):
            if counter > early_stopping_patience:
                epoch -= 1
                break
            self.lr = scheduler.get_rate(epoch)
            loss = 0.0
            for idx in range(X.shape[0]):
                x, label = X[idx], y[idx]
                self._backward(x, label)
                loss += self._loss(self.predict(x),label)
            mse = loss / X.shape[0]
            if mse > best_loss:
                counter += 1
            else:
                counter = 0
                best_loss = mse
            train_loss.append(mse)
            print(f'epoch {epoch}/{max_iter},train loss:{mse:4f},learing rate:{self.lr:2f}')
        print(f'finished!')
        return epoch,train_loss

    def save(self,root):
        '''
        save trained parameters
        :param root: path to save
        '''
        with open(f'{root}/weights','wb') as f:
            pkl.dump(self.weights,f)
        with open(f'{root}/biases', 'wb') as f:
            pkl.dump(self.biases,f)

    def load(self, root):
        '''
        load trained parameters
        :param root: path of save
        '''
        with open(f'{root}/weights', 'rb') as f:
            self.weights = pkl.load(f)
        with open(f'{root}/biases', 'rb') as f:
            self.biases = pkl.load(f)
