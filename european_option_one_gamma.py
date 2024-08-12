# -*- coding: utf-8 -*-
"""
European call option with one interest rate curve

@author: Marie Thibeau
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import timeit
import csv

#%% Some useful functions from the Nelson-Siegel model

def f_function(t,b0,b10,b11,c1):
    # forward rate, f(0,t)
    # modelised by the Nelson-Siegel model
    f = b0 + (b10 + b11*t)*np.exp(-c1*t)
    return f

def df_function(t,b0,b10,b11,c1):
    # derivative of forward rate, df(0,t)/dt
    df = (b11 - c1*(b10 + b11*t))*np.exp(-c1*t)
    return df

def r0(t,b0,b10,b11,c1):
    # initial zero-coupon yields
    # modelised by the Nelson-Siegel model
    t2 = 1/t * b10/c1 * (1-np.exp(-c1*t))
    t3 = 1/t * b11/c1**2 * (1-(c1*t+1)*np.exp(-c1*t))
    return b0 + t2 + t3

def P0(t,b0,b10,b11,c1):
    # price of a zero-coupon bond of maturity t
    out1 = np.zeros_like(t,dtype=float)
    if (type(t)==np.ndarray): 
        out1[(t!=0)] = r0(t[(t!=0)],b0,b10,b11,c1)  
    else:  
        if (t==0): 
            out1 = 0 
        else:
            out1 = r0(t,b0,b10,b11,c1)  
    out = np.exp(-out1*t)
    return out

def gamma(t,k,s_r,b0,b10,b11,c1):
    # mean-reverting phenomenon of rt 
    # fitted to term structure of interest rates
    g = ((1/k) * df_function(t,b0,b10,b11,c1) 
         + f_function(t,b0,b10,b11,c1) 
         + (s_r**2/(2*k**2)) * (1 - np.exp(-2*k*t)))
    return g

#%% Closed-from expression

def B(t,T,k):
    return (1 - np.exp(-k*(T-t)))/k

def m_square(t,T,k,s_r,s_S,r):
    # adapted variance
    t1 = s_S**2 * (T-t)
    t2 = s_r**2 * (1/(2*k**3))*(2*k*(T-t) + 4*np.exp(-k*(T-t)) - np.exp(-2*k*(T-t)) - 3)
    t3 = 2*r*s_r*s_S * (1/k**2)*(k*(T-t) + np.exp(-k*(T-t)) - 1)
    return t1 + t2 + t3

def P(t,T,rt,k,s_r,b0,b10,b11,c1):
    # fair value of the zero-coupon bond of maturity T at time t
    t1 = -rt*B(t,T,k) + f_function(t,b0,b10,b11,c1)*B(t,T,k) + np.log(P0(T,b0,b10,b11,c1)/P0(t,b0,b10,b11,c1))
    t2 = -s_r**2/(4*k**3) * (1-np.exp(-2*k*t)) * (B(t,T,k))**2
    return np.exp(t1) * np.exp(t2)

def F(t,T,rt,St,k,s_r,s_S,r,K,b0,b10,b11,c1):
    # fair value of the derivative of maturity T at time t
    # given by a Black & Scholes formula
    h1 = (np.log(St/K) - np.log(P(t,T,rt,k,s_r,b0,b10,b11,c1)) + 0.5*m_square(t,T,k,s_r,s_S,r)) / np.sqrt(m_square(t,T,k,s_r,s_S,r))
    h2 = (np.log(St/K) - np.log(P(t,T,rt,k,s_r,b0,b10,b11,c1)) - 0.5*m_square(t,T,k,s_r,s_S,r)) / np.sqrt(m_square(t,T,k,s_r,s_S,r))
    F = St*norm.cdf(h1) - K*P(t,T,rt,k,s_r,b0,b10,b11,c1)*norm.cdf(h2)
    return F

#%% Definition of the neural network model: 
    # FNN (feedforward neural network), 
    # RNN (residual neural network = FNN with skip connections) or 
    # DGM (neural network with Deep Galerkin Method layers)

class FNN(tf.keras.Model):
    def __init__(self,num_neurons,num_hidden_layers):
        super(FNN,self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.dense_layers = [tf.keras.layers.Dense(num_neurons, activation='tanh')]
        for _ in range(num_hidden_layers):
            self.dense_layers.append(tf.keras.layers.Dense(num_neurons, activation='tanh'))
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
        
    def call(self,inputs):
        t, rt, St, kappa, sigma_r, sigma_S, rho, T = inputs
        u = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho, T], axis=1)
        for layer in self.dense_layers[0:]:
            u = layer(u)
        u = self.output_layer(u)
        return u
    
class RNN(tf.keras.Model):
    def __init__(self,num_neurons,num_hidden_layers):
        super(RNN,self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.dense_layers = [tf.keras.layers.Dense(num_neurons, activation='tanh')]
        for _ in range(num_hidden_layers):
            self.dense_layers.append(tf.keras.layers.Dense(num_neurons, activation='tanh'))
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
        
    def call(self,inputs):
        t, rt, St, kappa, sigma_r, sigma_S, rho, T = inputs
        u0 = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho, T], axis=1)
        u = self.dense_layers[0](u0)
        for layer in self.dense_layers[1:]:
            u = layer(tf.concat([u,u0],axis=1))
        u = self.output_layer(u)
        return u
    
class DGM(tf.keras.Model):
    def __init__(self, num_neurons,num_hidden_layers):
        super(DGM, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)

        self.Sw  = tf.keras.layers.Dense(num_neurons)
        self.Uz  = tf.keras.layers.Dense(num_neurons)
        self.Wsz = tf.keras.layers.Dense(num_neurons)
        self.Ug  = tf.keras.layers.Dense(num_neurons)
        self.Wsg = tf.keras.layers.Dense(num_neurons)
        self.Ur  = tf.keras.layers.Dense(num_neurons)
        self.Wsr = tf.keras.layers.Dense(num_neurons)
        self.Uh  = tf.keras.layers.Dense(num_neurons)
        self.Wsh = tf.keras.layers.Dense(num_neurons)
        self.Wf  = tf.keras.layers.Dense(1)

    def call(self, inputs):
        t, rt, St, kappa, sigma_r, sigma_S, rho, T = inputs
        x    = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho, T], axis=1)
        S1 = self.Sw(x)
        out = S1
        for i in range(1, self.num_hidden_layers):
            S = self.sig_act(out)
            Z = self.sig_act(self.Uz(x) + self.Wsz(S))
            G = self.sig_act(self.Ug(x) + self.Wsg(S))
            R = self.sig_act(self.Ur(x) + self.Wsr(S))
            H = self.sig_act(self.Uh(x) + self.Wsh(S * R))
            out = (1 - G) * H + Z * S
        out = self.Wf(out)
        return out

#%% Definition of the loss function

def loss_function(model, t,  rt, St, kappa,  sigma_r,  sigma_S,  rho,  T,
                             rT, ST, kappaT, sigma_rT, sigma_ST, rhoT, TT, HT,
                         tl, rl, Sl, kappal, sigma_rl, sigma_Sl, rhol, Tl, Hl,
                         sc, ns, loss_weights):
    
    # conversion to tensors
    t_tf        = tf.convert_to_tensor(t, dtype=tf.float32)
    rt_tf       = tf.convert_to_tensor(rt, dtype=tf.float32)
    St_tf       = tf.convert_to_tensor(St, dtype=tf.float32)
    kappa_tf    = tf.convert_to_tensor(kappa, dtype=tf.float32)
    sigma_r_tf  = tf.convert_to_tensor(sigma_r, dtype=tf.float32)
    sigma_S_tf  = tf.convert_to_tensor(sigma_S, dtype=tf.float32)
    rho_tf      = tf.convert_to_tensor(rho, dtype=tf.float32)
    T_tf        = tf.convert_to_tensor(T, dtype=tf.float32)
    rT_tf       = tf.convert_to_tensor(rT, dtype=tf.float32)
    ST_tf       = tf.convert_to_tensor(ST, dtype=tf.float32)
    kappaT_tf   = tf.convert_to_tensor(kappaT, dtype=tf.float32)
    sigma_rT_tf = tf.convert_to_tensor(sigma_rT, dtype=tf.float32)
    sigma_ST_tf = tf.convert_to_tensor(sigma_ST, dtype=tf.float32)
    rhoT_tf     = tf.convert_to_tensor(rhoT, dtype=tf.float32)
    TT_tf       = tf.convert_to_tensor(TT, dtype=tf.float32)
    HT_tf       = tf.convert_to_tensor(HT, dtype=tf.float32)
    tl_tf       = tf.convert_to_tensor(tl, dtype=tf.float32)
    rl_tf       = tf.convert_to_tensor(rl, dtype=tf.float32)
    Sl_tf       = tf.convert_to_tensor(Sl, dtype=tf.float32)
    kappal_tf   = tf.convert_to_tensor(kappal, dtype=tf.float32)
    sigma_rl_tf = tf.convert_to_tensor(sigma_rl, dtype=tf.float32)
    sigma_Sl_tf = tf.convert_to_tensor(sigma_Sl, dtype=tf.float32)
    rhol_tf     = tf.convert_to_tensor(rhol, dtype=tf.float32)
    Tl_tf       = tf.convert_to_tensor(Tl, dtype=tf.float32)
    Hl_tf       = tf.convert_to_tensor(Hl, dtype=tf.float32)
    
    # calculation of gamma(t)
    gamma_tf = gamma(t_tf,kappa_tf,sigma_r_tf,ns['b0'],ns['b10'],ns['b11'],ns['c1'])
    
    # calculation of partial derivatives for scaled FK equation
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([rt_tf,St_tf,t_tf])
        u_pred = model([t_tf, rt_tf, St_tf, kappa_tf, sigma_r_tf, sigma_S_tf, rho_tf, T_tf])
        du   = tape1.gradient(u_pred, [rt_tf,St_tf,t_tf])
        u_r  = du[0]
        u_S  = du[1]
        u_t  = du[2]
        du2r = tape1.gradient(u_r, [rt_tf,St_tf])
        u_rr = du2r[0]
        u_rS = du2r[1]
        u_SS = tape1.gradient(u_S, St_tf)
    
    # inner loss
    residual_D = (sc['bh']*u_t - ((rt_tf-sc['ar'])/sc['br'])*u_pred
                  +kappa_tf*(sc['br']*gamma_tf+sc['ar']-rt_tf)*u_r + (St_tf-sc['aS']+((rt_tf-sc['ar'])/sc['br'])*u_S)
                  +0.5*(sc['br']**2*sigma_r_tf**2*u_rr + (St_tf-sc['aS'])**2*sigma_S_tf**2*u_SS) 
                  + sc['br']*sigma_r_tf*(St_tf-sc['aS'])*sigma_S_tf*rho_tf*u_rS)
    L_D = tf.reduce_mean(tf.square(residual_D))
    
    # maturity loss
    u_pred_T   = model([TT_tf, rT_tf, ST_tf, kappaT_tf, sigma_rT_tf, sigma_ST_tf, rhoT_tf, TT_tf])
    residual_T = u_pred_T - HT_tf
    L_T        = tf.reduce_mean(tf.square(residual_T))
    
    # lower loss
    u_pred_l   = model([tl_tf, rl_tf, Sl_tf, kappal_tf, sigma_rl_tf, sigma_Sl_tf, rhol_tf, Tl_tf])
    residual_l = u_pred_l - Hl_tf
    L_l        = tf.reduce_mean(tf.square(residual_l))
    
    # weighted loss and total loss
    weighted_loss = loss_weights[0]*L_D + loss_weights[1]*L_T + loss_weights[2]*L_l
    c = np.zeros_like(loss_weights)
    c[:] = loss_weights
    c[np.where(loss_weights==0)] = 1
    tot_loss_weights = loss_weights/c
    total_loss = tot_loss_weights[0]*L_D + tot_loss_weights[1]*L_T + tot_loss_weights[2]*L_l
    
    return weighted_loss, L_D, L_T, L_l, total_loss

#%% Definition of the training function

def train_function(model, t,  rt, St, kappa,  sigma_r,  sigma_S,  rho,  T,
                              rT, ST, kappaT, sigma_rT, sigma_ST, rhoT, TT, HT,
                          tl, rl, Sl, kappal, sigma_rl, sigma_Sl, rhol, Tl, Hl,
                          sc, ns, loss_weights, epochs=10000, lr=0.001, batch_perc=0.20):
    
    # optimizer choice
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # batch calculation
    num_samples = len(rt)
    batch_size  = int(np.round(batch_perc * num_samples, decimals=0))
    num_batches = int(num_samples // batch_size)  # inner, D
    num_sampleT = len(rT)
    batch_sizeT = num_sampleT // num_batches  # maturity boundary, T
    num_samplel = len(rl)
    batch_sizel = num_samplel // num_batches  # lower boundary, l
    
    # creation of a loss table
    losses = np.zeros((epochs, 5), dtype=float)
    
    for epoch in range(epochs):
        # shuffle the data indices for each epoch
        indices  = np.random.permutation(num_samples)
        indicesT = np.random.permutation(num_sampleT)
        indicesl = np.random.permutation(num_samplel)
        weight_loss  = 0.0
        total_loss_D = 0.0
        total_loss_T = 0.0
        total_loss_l = 0.0
        total_loss   = 0.0

        for batch in range(num_batches):
            # indices in the batch
            batch_indices  = indices[batch * batch_size: (batch + 1) * batch_size]
            batch_indicesT = indicesT[batch * batch_sizeT: (batch + 1) * batch_sizeT]
            batch_indicesl = indicesl[batch * batch_sizel: (batch + 1) * batch_sizel]
            
            # inputs for L_D
            t_batch       = t[batch_indices]
            rt_batch      = rt[batch_indices]
            St_batch      = St[batch_indices]
            kappa_batch   = kappa[batch_indices]
            sigma_r_batch = sigma_r[batch_indices]
            sigma_S_batch = sigma_S[batch_indices]
            rho_batch     = rho[batch_indices]
            T_batch       = T[batch_indices]

            # inputs for L_T
            rT_batch       = rT[batch_indicesT]
            ST_batch       = ST[batch_indicesT]
            kappaT_batch   = kappaT[batch_indicesT]
            sigma_rT_batch = sigma_rT[batch_indicesT]
            sigma_ST_batch = sigma_ST[batch_indicesT]
            rhoT_batch     = rhoT[batch_indicesT]
            TT_batch       = TT[batch_indicesT]
            HT_batch       = HT[batch_indicesT]
            
            # inputs for L_l
            tl_batch       = tl[batch_indicesl]
            rl_batch       = rl[batch_indicesl]
            Sl_batch       = Sl[batch_indicesl]
            kappal_batch   = kappal[batch_indicesl]
            sigma_rl_batch = sigma_rl[batch_indicesl]
            sigma_Sl_batch = sigma_Sl[batch_indicesl]
            rhol_batch     = rhol[batch_indicesl]
            Tl_batch       = Tl[batch_indicesl]
            Hl_batch       = Hl[batch_indicesl]
            
            # loss calculation
            with tf.GradientTape() as tape:
                # computation of L_weighted, L_D, L_T, L_l, L_tot
                loss, L_D, L_T, L_l, L_tot = loss_function(model, 
                                                           t_batch,  rt_batch, St_batch, kappa_batch,  sigma_r_batch,  sigma_S_batch,  rho_batch,  T_batch,
                                                                     rT_batch, ST_batch, kappaT_batch, sigma_rT_batch, sigma_ST_batch, rhoT_batch, TT_batch, HT_batch,
                                                           tl_batch, rl_batch, Sl_batch, kappal_batch, sigma_rl_batch, sigma_Sl_batch, rhol_batch, Tl_batch, Hl_batch,
                                                           sc, ns, loss_weights)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            weight_loss  += loss
            total_loss_D += L_D
            total_loss_T += L_T
            total_loss_l += L_l
            total_loss   += L_tot
        avg_loss = weight_loss / num_batches
        losses[epoch, :] = [avg_loss, total_loss_D / num_batches, total_loss_T / num_batches, 
                            total_loss_l / num_batches, total_loss / num_batches]
        
        # to see the progress of the calibration
        if (epoch + 1) % 100 == 0:
            print("Epoch {}/{}: Loss = {}".format(epoch + 1, epochs, avg_loss.numpy()))
            
    return losses

#%% Main

#%% Definition of parameters and range values

# Nelson-Siegel parameters
b0  =  0.03974889670677964
b10 =  0.0014721430763244212
b11 = -0.01693616284948873
c1  =  0.4222897254760573

# Lower and upper values of parameters and state varaibles
r_l       =  0.005; r_u       = 0.07 # stochastic interest rate
S_l       =  20;    S_u       = 250  # stock value
kappa_l   =  0.5;   kappa_u   = 2    # mean-reverting phenomenon of rt
sigma_r_l =  0.002; sigma_r_u = 0.02 # standard deviation of rt
sigma_S_l =  0.02;  sigma_S_u = 0.3  # standard deviation of St
rho_l     = -0.8;   rho_u     = 0.8  # correlation between rt and St
T_max     =  10

# Fixed parameters
K = 100                              # strike price of the option

# Hyperparameters of the neural network
NN_type = "DGM"                      # architecture of the neural network: 
                                         # FNN (feedforward neural network), 
                                         # RNN (residual neural network = FNN with skip connections) or 
                                         # DGM (neural network with Deep Galerkin Method layers)
num_hidden_layers = 3                # number of hidden layers in the network
num_neurons = 256                    # number of neurons by hidden layer

# Sample sets size for training the neural network
num_samples = 20000                  # number of samples for the inner loss
num_samples_bound = 5000             # number of samples for the maturity loss and lower loss
# Sample set size for vizualizing the result
num_samples_test = 50

# Loss weights in the total loss
inner_loss_weight    = 10
maturity_loss_weight = 10
lower_loss_weight    = 1
loss_weights = np.array([inner_loss_weight,maturity_loss_weight,lower_loss_weight])

# Learning rates and epochs
lr_1 = 0.005;  epochs_1 = 500
lr_2 = 0.002;  epochs_2 = 1000
lr_3 = 0.001;  epochs_3 = 1000
lr_4 = 0.0001; epochs_4 = 1000

# Directory and file names for saving weights, scaling and results
Directory_name = "EU_one_gamma__"+NN_type+"_"+str(num_hidden_layers)+"_"+str(num_neurons)+"__"+str(num_samples)+"_"+str(num_samples_bound)+"__"+str(inner_loss_weight)+"_"+str(maturity_loss_weight)+"_"+str(lower_loss_weight)
exists = os.path.exists(Directory_name)
if not exists:
    os.mkdir(Directory_name)
weight_name     = "weights__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".h5"
scaling_ay_name = "scaling_ay__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".npy"
scaling_by_name = "scaling_by__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".npy"
plots_loss_name = "plots_loss__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
plot_singleset_rt_name = "plot_single_set_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
plot_singleset_St_name = "plot_single_set_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
heatmap_alea_rt_name = "heatmap_alea_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
heatmap_alea_St_name = "heatmap_alea_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
heatmap_conf_rt_name = "heatmap_conf_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"
heatmap_conf_St_name = "heatmap_conf_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".png"

#%% Construction of sample sets for training

# for reproductability
np.random.seed(17)

# Inner sample set
rt      = np.random.uniform(r_l,r_u,num_samples).reshape(-1, 1)
St      = np.random.uniform(S_l,S_u,num_samples).reshape(-1, 1)
kappa   = np.random.uniform(kappa_l,kappa_u,num_samples).reshape(-1, 1)
sigma_r = np.random.uniform(sigma_r_l,sigma_r_u,num_samples).reshape(-1, 1)
sigma_S = np.random.uniform(sigma_S_l,sigma_S_u,num_samples).reshape(-1, 1)
rho     = np.random.uniform(rho_l,rho_u,num_samples).reshape(-1, 1)
T       = np.random.uniform(0,T_max,num_samples).reshape(-1, 1)
t       = np.zeros_like(T)
for i in range(num_samples):
    t[i] = np.random.uniform(0,T[i],1).reshape(-1, 1)
# scaling of state variables
rbar = np.mean(rt); Sr = np.std(rt)
Sbar = np.mean(St); SS = np.std(St)
ar = -rbar/Sr; br = 1/Sr 
aS = -Sbar/SS; bS = 1/SS
ah = -1/2;     bh = 1/T_max
ay = np.array((ar,aS))
by = np.array((br,bS))

rt_tilde = ar+br*rt
St_tilde = aS+bS*St
t_tilde  = ah+bh*t
T_tilde  = ah+bh*T

# Maturity sample set
rT       = np.random.uniform(r_l,r_u,num_samples_bound).reshape(-1, 1)
ST       = np.random.uniform(S_l,S_u,num_samples_bound).reshape(-1, 1)
kappaT   = np.random.uniform(kappa_l,kappa_u,num_samples_bound).reshape(-1, 1)
sigma_rT = np.random.uniform(sigma_r_l,sigma_r_u,num_samples_bound).reshape(-1, 1)
sigma_ST = np.random.uniform(sigma_S_l,sigma_S_u,num_samples_bound).reshape(-1, 1)
rhoT     = np.random.uniform(rho_l,rho_u,num_samples_bound).reshape(-1, 1)
TT       = np.random.uniform(0,T_max,num_samples_bound).reshape(-1, 1)
# scaling of state variables
rT_tilde = ar+br*rT
ST_tilde = aS+bS*ST
TT_tilde = ah+bh*TT
# payoff at maturity
HT = np.maximum(ST-K,0)

# Lower sample set
rl       = np.random.uniform(r_l,r_u,num_samples_bound).reshape(-1, 1)
Sl       = np.zeros(rl.shape); Sl[:] = 0.0
kappal   = np.random.uniform(kappa_l,kappa_u,num_samples_bound).reshape(-1, 1)
sigma_rl = np.random.uniform(sigma_r_l,sigma_r_u,num_samples_bound).reshape(-1, 1)
sigma_Sl = np.random.uniform(sigma_S_l,sigma_S_u,num_samples_bound).reshape(-1, 1)
rhol     = np.random.uniform(rho_l,rho_u,num_samples_bound).reshape(-1, 1)
Tl       = np.random.uniform(0,T_max,num_samples_bound).reshape(-1, 1)
tl       = np.zeros_like(Tl)
for i in range(num_samples_bound):
    tl[i] = np.random.uniform(0,Tl[i],1).reshape(-1, 1)
# scaling of state variables
rl_tilde = ar+br*rl
Sl_tilde = aS+bS*Sl
tl_tilde = ah+bh*tl
Tl_tilde = ah+bh*Tl
# discounted payoff at maturity when St=0
Hl = np.zeros(rl.shape); Hl[:] = 0.0

#%% Training of a new model (TRAIN=True) or loading of weights (TRAIN=False)

# dictionnary for scaling parameters and Nelson-Siegel parameters
sc = {'ar':ar, 'br':br, 'aS':aS, 'bS':bS, 'ah':ah, 'bh':bh}
ns = {'b0':b0, 'b10':b10, 'b11':b11, 'c1':c1}

# model initialization
if NN_type=="FNN":
    model = FNN(num_neurons, num_hidden_layers)
elif NN_type=="RNN":
    model = RNN(num_neurons, num_hidden_layers)
elif NN_type=="DGM":
    model = DGM(num_neurons, num_hidden_layers)
    

TRAIN = False

if TRAIN:
    tic = timeit.default_timer()
    
    loss_tab_1 = train_function(model, 
                                t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho,  T_tilde,
                                          rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, TT_tilde, HT,
                                tl_tilde, rl_tilde, Sl_tilde, kappal, sigma_rl, sigma_Sl, rhol, Tl_tilde, Hl,
                                sc, ns, loss_weights, epochs_1, lr_1, batch_perc=0.34)

    loss_tab_2 = train_function(model, 
                                t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho,  T_tilde,
                                          rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, TT_tilde, HT,
                                tl_tilde, rl_tilde, Sl_tilde, kappal, sigma_rl, sigma_Sl, rhol, Tl_tilde, Hl,
                                sc, ns, loss_weights, epochs_2, lr_2, batch_perc=0.34)

    loss_tab_3 = train_function(model, 
                                t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho,  T_tilde,
                                          rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, TT_tilde, HT,
                                tl_tilde, rl_tilde, Sl_tilde, kappal, sigma_rl, sigma_Sl, rhol, Tl_tilde, Hl,
                                sc, ns, loss_weights, epochs_3, lr_3, batch_perc=0.34)

    loss_tab_4 = train_function(model, 
                                t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho,  T_tilde,
                                          rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, TT_tilde, HT,
                                tl_tilde, rl_tilde, Sl_tilde, kappal, sigma_rl, sigma_Sl, rhol, Tl_tilde, Hl,
                                sc, ns, loss_weights, epochs_4, lr_4, batch_perc=0.34)
    
    toc = timeit.default_timer()
    train_time= toc-tic
    print('Training time:', round(train_time,1) , 'sec')
    model.count_params()
    model.save_weights(Directory_name +"/" + weight_name)    
    np.save(Directory_name +"/" + scaling_ay_name,ay)
    np.save(Directory_name +"/" + scaling_by_name,by)

    fig, ax = plt.subplots(2, 2)
    epochs = epochs_1 + epochs_2 + epochs_3 + epochs_4
    loss_tab = np.concatenate((loss_tab_1,loss_tab_2,loss_tab_3,loss_tab_4))
    plt.subplot(2, 2, 1)
    ax[0, 0].plot(np.arange(0,epochs), loss_tab[:,0])
    plt.subplot(2, 2, 2)
    ax[0, 1].plot(np.arange(0,epochs),loss_tab[:,1])
    plt.subplot(2, 2, 3)
    ax[1, 0].plot(np.arange(0,epochs),loss_tab[:,2])
    plt.subplot(2, 2, 4)
    ax[1, 1].plot(np.arange(0,epochs),loss_tab[:,3])

    ax[0, 0].set_title("Weighted Loss")
    ax[0, 1].set_title("Inner Loss")
    ax[1, 0].set_title("Maturity Loss")
    ax[1, 1].set_title("Lower Loss")

    fig.tight_layout()
    fig.savefig(Directory_name +"/" + plots_loss_name)
    plt.show()

    print("Weighted Loss:", loss_tab[:,0][-1])
    print("Inner Loss:", loss_tab[:,1][-1])
    print("Maturity Loss:", loss_tab[:,2][-1])
    print("Lower Loss:", loss_tab[:,3][-1])
    print("Total Loss:", loss_tab[:,4][-1])

else :
    # initializing the neural network (tmp not used after)
    tmp = model([t_tilde, rt_tilde, St_tilde, kappa, sigma_r, sigma_S, rho, T_tilde])
    # loading the calibrated weights
    model.load_weights(Directory_name +"/" +weight_name)
    # loading the scaling parameters
    ay = np.load(Directory_name +"/" +scaling_ay_name)
    by = np.load(Directory_name +"/" +scaling_by_name)
    ar = ay[0]; aS = ay[1]
    br = by[0]; bS = by[1]

    # running only 10 epochs for getting the losses
    loss_tab = train_function(model, 
                                t_tilde, rt_tilde, St_tilde, kappa, sigma_r, sigma_S, rho, T_tilde,
                                         rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, TT_tilde,  HT,
                                tl_tilde, rl_tilde, Sl_tilde, kappal, sigma_rl, sigma_Sl, rhol, Tl_tilde,  Hl,
                                sc, ns, loss_weights, 10, lr_4, batch_perc=0.34)
    train_time = 0
    
#%% Validation: comparison with closed-form expression

#%% u_pred vs. rt with a single set of parameters

# state variables and parameters
rt_test      = np.linspace(0.005, 0.07, num_samples_test).reshape(-1, 1)
St_test      = np.zeros_like(rt_test).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
T_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
St_test[:]      =  110
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4 
t_test[:]       =  0
T_test[:]       =  5 
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test])

# fair value given by the closed-form formula
F_cf = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)

# plot of PINN & closed-form prices
plt.plot(rt_test,F_cf, "-g", label="Exact price")
plt.plot(rt_test, F_NN, "-b", label = "PINN price")
# plt.plot(rt_test, np.abs(F_NN - F_cf), "-r", label = "absolute difference")
plt.legend(loc='lower right')
plt.xlabel('$r_t$')
plt.ylabel('$F_t$')
plt.savefig(Directory_name +"/" + plot_singleset_rt_name, bbox_inches='tight')
plt.show()

#%% u_pred vs. St with a single set of parameters

# state variables and parameters
St_test      = np.linspace(50, 150, num_samples_test).reshape(-1, 1)
rt_test      = np.zeros_like(St_test).reshape(-1, 1)
kappa_test   = np.zeros_like(St_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(St_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(St_test).reshape(-1, 1)
rho_test     = np.zeros_like(St_test).reshape(-1, 1)
t_test       = np.zeros_like(St_test).reshape(-1, 1)
T_test       = np.zeros_like(St_test).reshape(-1, 1)
K_test       = np.zeros_like(St_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4 
t_test[:]       =  0
T_test[:]       =  5 
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test])

# fair value given by the closed-form formula
F_cf = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)

# plot of PINN & closed-form prices
plt.plot(St_test, np.maximum(St_test-K_test,np.zeros_like(St_test)), "--k", label="Payoff")
plt.plot(St_test,F_cf, "-g", label="Exact price")
plt.plot(St_test, F_NN, "-b", label = "PINN price")
# plt.plot(St_test, F_NN - F_cf, "-r", label = "difference")
plt.legend(loc='upper left')
plt.xlabel('$S_t$')
plt.ylabel('$F_t$')
plt.savefig(Directory_name +"/" + plot_singleset_St_name , bbox_inches='tight')
plt.show()

#%% fixed rt and T, rest random

np.random.seed(18)
# random sample test
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.random.uniform(S_l, S_u, num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
a = list(np.linspace(0.01, T_max, num_samples_test))
T_test = []
for i in range(len(a)):
  T_test.append([a[i] for j in range(num_samples_test)])
T_test = np.array(T_test).reshape(-1, 1)
t_test = np.zeros_like(St_test).reshape(-1, 1)
t_test[:] = 0
K_test    = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:] = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN       = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test )

# fair value given by the closed-form formula
F_cf       = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf       = F_cf.reshape(-1, 1)
F_cf_array = np.array(F_cf).reshape(num_samples_test, num_samples_test)

# difference
diff       = np.abs(F_NN - F_cf)
diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)

# plot of PINN & closed-form prices: heatmap rt/T    
df = pd.DataFrame(diff_array, index = [T_max*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$r_t$ [%]')
plt.ylabel('$T$ [year]')
plt.savefig(Directory_name +"/" + heatmap_alea_rt_name , bbox_inches='tight')
plt.show()

# MSE calculation
MSE_alea_rt = F_NN - F_cf
MSE_alea_rt = np.mean(np.square(MSE_alea_rt[~np.isnan(MSE_alea_rt)]))
print("Mean Square Error:", MSE_alea_rt)

#%% fixed St and T, rest random

np.random.seed(19)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, num_samples_test**2).reshape(-1, 1)
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
a = list(np.linspace(0.01, T_max, num_samples_test))
T_test = []
for i in range(len(a)):
  T_test.append([a[i] for j in range(num_samples_test)])
T_test = np.array(T_test).reshape(-1, 1)
t_test = np.zeros_like(St_test).reshape(-1, 1)
t_test[:] = 0
K_test    = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:] = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN       = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test )

# fair value given by the closed-form formula
F_cf       = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf       = F_cf.reshape(-1, 1)
F_cf_array = np.array(F_cf).reshape(num_samples_test, num_samples_test)

# difference
diff       = np.abs(F_NN - F_cf)
diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)

# plot of PINN & closed-form prices: heatmap St/T
df = pd.DataFrame(diff_array, index = [T_max*i/num_samples_test for i in range(0,num_samples_test)], columns = [int(S_l + (S_u - S_l)*(i-1)/num_samples_test) for i in range(1,num_samples_test + 1)])
sns.heatmap(df,cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$S_t$')
plt.ylabel('$T$ [year]')
plt.savefig(Directory_name +"/" + heatmap_alea_St_name , bbox_inches='tight')
plt.show()

# MSE calculation
MSE_alea_St = F_NN - F_cf
MSE_alea_St = np.mean(np.square(MSE_alea_St[~np.isnan(MSE_alea_St)]))
print("Mean Square Error:", MSE_alea_St)

#%% fixed rt and T, rest in a particular configuration

# state variables and parameters
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.zeros_like(rt_test).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
a = list(np.linspace(0.01, T_max, num_samples_test))
T_test = []
for i in range(len(a)):
  T_test.append([a[i] for j in range(num_samples_test)])
T_test = np.array(T_test).reshape(-1, 1)
K_test = np.zeros_like(rt_test).reshape(-1, 1)
# values
St_test[:]      =  110
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4 
t_test[:]       =  0
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN       = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test )

# fair value given by the closed-form formula
F_cf       = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf       = F_cf.reshape(-1, 1)
F_cf_array = np.array(F_cf).reshape(num_samples_test, num_samples_test)

# difference
diff       = np.abs(F_NN - F_cf)
diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)

# plot of PINN & closed-form prices: heatmap rt/T
df = pd.DataFrame(diff_array, index = [T_max*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df,cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$r_t$ [%]')
plt.ylabel('$T$ [year]')
plt.savefig(Directory_name +"/" + heatmap_conf_rt_name, bbox_inches='tight')
plt.show()

# MSE calculation
MSE_conf_rt = F_NN - F_cf
MSE_conf_rt = np.mean(np.square(MSE_conf_rt[~np.isnan(MSE_conf_rt)]))
print("Mean Square Error:", MSE_conf_rt)

#%% fixed St and T, rest in a particular configuration

# state variables and parameters
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
rt_test      = np.zeros_like(St_test).reshape(-1, 1)
kappa_test   = np.zeros_like(St_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(St_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(St_test).reshape(-1, 1)
rho_test     = np.zeros_like(St_test).reshape(-1, 1)
t_test       = np.zeros_like(St_test).reshape(-1, 1)
a = list(np.linspace(0.01, T_max, num_samples_test))
T_test = []
for i in range(len(a)):
  T_test.append([a[i] for j in range(num_samples_test)])
T_test = np.array(T_test).reshape(-1, 1)
K_test = np.zeros_like(St_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4 
t_test[:]       =  0
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN       = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test )

# fair value given by the closed-form formula
F_cf       = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf       = F_cf.reshape(-1, 1)
F_cf_array = np.array(F_cf).reshape(num_samples_test, num_samples_test)

# difference
diff       = np.abs(F_NN - F_cf)
diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)

# plot of PINN & closed-form prices: heatmap St/T
df = pd.DataFrame(diff_array, index = [T_max*i/num_samples_test for i in range(0,num_samples_test)], columns = [int(S_l + (S_u - S_l)*(i-1)/num_samples_test) for i in range(1,num_samples_test + 1)])
sns.heatmap(df,cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$S_t$')
plt.ylabel('$T$ [year]')
plt.savefig(Directory_name +"/" + heatmap_conf_St_name, bbox_inches='tight')
plt.show()

# MSE calculation
MSE_conf_St = F_NN - F_cf
MSE_conf_St = np.mean(np.square(MSE_conf_St[~np.isnan(MSE_conf_St)]))
print("Mean Square Error:", MSE_conf_St)

#%% all random

np.random.seed(20)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(S_l, S_u, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, 2*num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, 2*num_samples_test**2).reshape(-1, 1)
T_test       = np.random.uniform(0, T_max, 2*num_samples_test**2).reshape(-1, 1)
t_test = np.zeros_like(rt_test).reshape(-1, 1)
for i in range(2*num_samples_test**2):
    t_test[i]  = np.random.uniform(0 , T_test[i] , 1).reshape(-1, 1)
K_test    = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:] = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])

# fair value given by the closed-form formula
F_cf = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf = F_cf.reshape(-1, 1)

# MSE calculation
MSE_alea = F_NN - F_cf
len_alea = len(MSE_alea[~np.isnan(MSE_alea)])
MSE_alea = np.mean(np.square(MSE_alea[~np.isnan(MSE_alea)]))
print("Mean Square Error:", MSE_alea)

#%% state variables rt and St random, rest in a particular configuration

np.random.seed(21)
# random sample test for state variables
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(S_l, S_u, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
T_test       = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4 
t_test[:]       =  0 
T_test[:]       =  5 
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])

# fair value given by the closed-form formula
F_cf = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf = F_cf.reshape(-1, 1)

# MSE calculation
MSE_conf = F_NN - F_cf
len_conf = len(MSE_conf[~np.isnan(MSE_conf)])
MSE_conf = np.mean(np.square(MSE_conf[~np.isnan(MSE_conf)]))
print("Mean Square Error:", MSE_conf)

#%% relative error in the money (St>K), all random

np.random.seed(22)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(K+1, S_u, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, 2*num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, 2*num_samples_test**2).reshape(-1, 1)
T_test       = np.random.uniform(0, T_max, 2*num_samples_test**2).reshape(-1, 1)
t_test = np.zeros_like(rt_test).reshape(-1, 1)
for i in range(2*num_samples_test**2):
    t_test[i]  = np.random.uniform(0 , T_test[i] , 1).reshape(-1, 1)
K_test    = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:] = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test
t_tilde_test  = ah+bh*t_test
T_tilde_test  = ah+bh*T_test

# fair value given by the neural network
F_NN = model([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, T_tilde_test ])

# fair value given by the closed-form formula
F_cf = F(t_test, T_test, rt_test, St_test, kappa_test, sigma_r_test, sigma_S_test, rho_test, K_test, b0, b10, b11, c1)
F_cf = F_cf.reshape(-1, 1)

# relative error calculation
err  = np.abs(F_NN - F_cf)
err  = err[~np.isnan(F_cf)]
F_cf = F_cf[~np.isnan(F_cf)]
err  = err[np.nonzero(F_cf)]
F_cf = F_cf[np.nonzero(F_cf)]
err_relative = np.mean(err/np.abs(F_cf))
print("Relative error in the money: ", err_relative)

#%% Saving outputs in a csv file

MSE = [MSE_alea_rt, MSE_alea_St, MSE_conf_rt, MSE_conf_St,  MSE_alea, MSE_conf,  err_relative]

exists_resloc = os.path.exists(Directory_name +"/" +'results.csv')
if not exists_resloc:
    with open(Directory_name +"/" +'results.csv', 'w', newline='') as file:
         writer = csv.writer(file, delimiter=';')
         writer.writerow(["Configuration","Weighted Loss", "Inner Loss", "Maturity Loss", "Lower Loss", "Total Loss",
                          "MSE_alea_rt", "MSE_alea_St", "MSE_conf_rt", "MSE_conf_St", "MSE_alea" , 
                          "MSE_conf",  "relative error (in the money)", "Training time [sec]", "TRAIN"])
         writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(train_time), TRAIN])
else:
    with open(Directory_name +"/" +'results.csv', 'a', newline='') as file:
         writer = csv.writer(file, delimiter=';')
         writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(train_time), TRAIN])
         
#%% Saving outputs in a csv file countaning outputs of all neural networks

exists_resall = os.path.exists('EU_results_all.csv')
if not exists_resall:
    with open('EU_results_all.csv', 'w', newline='') as file:
         writer = csv.writer(file, delimiter=';')
         writer.writerow(["Configuration","Weighted Loss", "Inner Loss", "Maturity Loss", "Lower Loss", "Total Loss",
                          "MSE_alea_rt", "MSE_alea_St", "MSE_conf_rt", "MSE_conf_St", "MSE_alea" , 
                          "MSE_conf",  "relative error (in the money)", "Training time [sec]", "TRAIN"])
         writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(train_time), TRAIN])
else:
    with open('EU_results_all.csv', 'a', newline='') as file:
         writer = csv.writer(file, delimiter=';')
         writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(train_time), TRAIN])
