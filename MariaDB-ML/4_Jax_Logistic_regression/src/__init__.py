import jax
import jax_metrics as jm
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import numpy as np
import matplotlib.pyplot as plt
# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

def test():
    print("ok")

class Logistic_Regression():
    """
    Basic Model + Quasi Newton Methods
    """
    def __init__(self, regularization='l2', method_opt='classic_model'):
        self.regularization = regularization
        self.method_opt = method_opt
        self.error_gradient = 0.001
        self.key = random.PRNGKey(0)
        # You need to add some variables
        self.W = None

    @staticmethod
    @jit
    def logistic_exp(W:jnp, X:jnp)->jnp:
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d + 1
            X is a d x N
        """
        return jnp.exp(W@X)

    @staticmethod
    @jit
    def logistic_sum(exTerms: jnp)->jnp:        
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d 
            X is a d x N
        """
        temp = jnp.sum(exTerms, axis=0)
        n = temp.shape[0]
        return jnp.reshape(1.0+temp, newshape=(1, n))

    @staticmethod
    @jit
    def logit_matrix(Terms: jnp, sum_terms: jnp)->jnp:
        """
        Generate matrix
        """
        divisor = 1/sum_terms
        n, _ = Terms.shape
        replicate = jnp.repeat(divisor, repeats=n, axis=0 )
        logits = Terms*replicate
        return jnp.vstack([logits, divisor])
    
    @partial(jit, static_argnums=(0,))
    def model(self, W:jnp, X:jnp, Y_hot:jnp)->jnp:
        """
        Logistic Model
        """
        W = jnp.reshape(W, self.sh)
        terms = self.logistic_exp(W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.sum(jnp.sum(jnp.log(matrix)*Y_hot, axis=0), axis=0)
    
    @staticmethod
    def one_hot(Y: jnp):
        """
        One_hot matrix
        """
        numclasses = len(jnp.unique(Y))
        return jnp.transpose(jax.nn.one_hot(Y, num_classes=numclasses))
    
    def generate_w(self, k_classes:int, dim:int)->jnp:
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        key = random.PRNGKey(0)
        keys = random.split(key, 1)
        return jnp.array(random.normal(keys[0], (k_classes, dim)))

    @staticmethod
    def augment_x(X: jnp)->jnp:
        """
        Augmenting samples of a dim x N matrix
        """
        N = X.shape[1]
        return jnp.vstack([X, jnp.ones((1, N))])
     
   
    def fit(self, X: jnp, Y:jnp)->None:
        """
        The fit process
        """
        nclasses = len(jnp.unique(Y))
        X = self.augment_x(X)
        dim = X.shape[0]
        W = self.generate_w(nclasses-1, dim)
        Y_hot = self.one_hot(Y)
        self.W = getattr(self, self.method_opt, lambda W, X, Y_hot: self.error() )(W, X, Y_hot)
    
    @staticmethod
    def error()->None:
        """
        Only Print Error
        """
        raise Exception("Opt Method does not exist")
    
    def classic_model(self, W:jnp, X:jnp, Y_hot:jnp, alpha:float=1e-2,  tol:float=1e-3)->jnp:
        """
        The naive version of the logistic regression
        """
        n, m = W.shape 
        self.sh = (n, m)
        alpha = 0.5
        Grad = jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)
        loss = self.model(jnp.ravel(W), X, Y_hot)
        cnt = 0
        while True:
            Hessian = jax.hessian(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)
            W = W - alpha*jnp.reshape((jnp.linalg.inv(Hessian)@Grad), self.sh)
            Grad =  jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot)
            old_loss = loss
            loss = self.model(jnp.ravel(W), X, Y_hot)
            if cnt%30 == 0:
                print(f'{self.model(jnp.ravel(W), X, Y_hot)}')
            if  jnp.abs(old_loss - loss) < tol:
                break
            cnt +=1
        return W
    
    def estimate_prob(self, X:jnp)->jnp:
        """
        Estimate Probability
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return matrix
    
    def estimate(self, X:jnp)->jnp:
        """
        Estimation
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.argmax(matrix, axis=0)
    
    def precision(self, y, y_hat):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP = sum(y_hat == y)
        FP = sum(y_hat != y)
        return (TP/(TP+FP)).tolist()
    
class Tools():
    """
    Tools
    """
    def __init__(self):
        """
        Basic Init
        """
        self.key = random.PRNGKey(0)
    
    def GenerateData(self, n_samples: int, n_classes: int, dim: int)-> (jnp, jnp):
        """
        Data Generation
        """
        Total_Data = [] 
        Total_Y = []
        for idx in range(n_classes):
            keys = random.split(self.key, 1)
            X = random.normal(keys[0], (dim, n_samples)) + idx*5*jnp.ones((dim, 1))
            Y = idx*jnp.ones(n_samples)
            Total_Data.append(X)
            Total_Y.append(Y)
        return jnp.hstack(Total_Data), jnp.hstack(Total_Y)
    
    @staticmethod
    def plot_classes(X: jnp, Y: jnp, n_classes: int)-> None:
        """
        Plot the classes
        """
        symbols = ['ro', 'bx', 'go', 'rx']
        plt.figure()
        for idx in range(n_classes):
            mask = idx == Y
            X_p = X[:, mask]
            plt.plot(X_p[0,:], X_p[1,:], symbols[idx])
        