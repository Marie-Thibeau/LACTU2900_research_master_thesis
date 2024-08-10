# -*- coding: utf-8 -*-
"""
Bermudan option with one interest rate curve

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
        t, rt, St, kappa, sigma_r, sigma_S, rho = inputs
        u = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho], axis=1)
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
        t, rt, St, kappa, sigma_r, sigma_S, rho = inputs
        u0 = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho], axis=1)
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
        t, rt, St, kappa, sigma_r, sigma_S, rho = inputs
        x   = tf.concat([t, rt, St, kappa, sigma_r, sigma_S, rho], axis=1)
        S1  = self.Sw(x)
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

def loss_function(model, t,  rt, St, kappa,  sigma_r,  sigma_S,  rho,
                             rT, ST, kappaT, sigma_rT, sigma_ST, rhoT, HT,
                         te, re, Se, kappae, sigma_re, sigma_Se, rhoe, He,
                         T, sc, ns, loss_weights):
    
    # conversion to tensors
    t_tf        = tf.convert_to_tensor(t, dtype=tf.float32)
    rt_tf       = tf.convert_to_tensor(rt, dtype=tf.float32)
    St_tf       = tf.convert_to_tensor(St, dtype=tf.float32)
    kappa_tf    = tf.convert_to_tensor(kappa, dtype=tf.float32)
    sigma_r_tf  = tf.convert_to_tensor(sigma_r, dtype=tf.float32)
    sigma_S_tf  = tf.convert_to_tensor(sigma_S, dtype=tf.float32)
    rho_tf      = tf.convert_to_tensor(rho, dtype=tf.float32)
    T_tf        = tf.convert_to_tensor(T*np.ones_like(rT), dtype=tf.float32)
    rT_tf       = tf.convert_to_tensor(rT, dtype=tf.float32)
    ST_tf       = tf.convert_to_tensor(ST, dtype=tf.float32)
    kappaT_tf   = tf.convert_to_tensor(kappaT, dtype=tf.float32)
    sigma_rT_tf = tf.convert_to_tensor(sigma_rT, dtype=tf.float32)
    sigma_ST_tf = tf.convert_to_tensor(sigma_ST, dtype=tf.float32)
    rhoT_tf     = tf.convert_to_tensor(rhoT, dtype=tf.float32)
    HT_tf       = tf.convert_to_tensor(HT, dtype=tf.float32)
    te_tf       = tf.convert_to_tensor(te, dtype=tf.float32)
    re_tf       = tf.convert_to_tensor(re, dtype=tf.float32)
    Se_tf       = tf.convert_to_tensor(Se, dtype=tf.float32)
    kappae_tf   = tf.convert_to_tensor(kappae, dtype=tf.float32)
    sigma_re_tf = tf.convert_to_tensor(sigma_re, dtype=tf.float32)
    sigma_Se_tf = tf.convert_to_tensor(sigma_Se, dtype=tf.float32)
    rhoe_tf     = tf.convert_to_tensor(rhoe, dtype=tf.float32)
    He_tf       = tf.convert_to_tensor(He, dtype=tf.float32)
    
    # calculation of gamma(t)
    gamma_tf = gamma(t_tf,kappa_tf,sigma_r_tf,ns['b0'],ns['b10'],ns['b11'],ns['c1'])
    
    # calculation of partial derivatives for scaled FK equation
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([rt_tf,St_tf,t_tf])
        u_pred = model([t_tf, rt_tf, St_tf, kappa_tf, sigma_r_tf, sigma_S_tf, rho_tf])
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
    u_pred_T   = model([T_tf, rT_tf, ST_tf, kappaT_tf, sigma_rT_tf, sigma_ST_tf, rhoT_tf])
    residual_T = u_pred_T - HT_tf
    L_T        = tf.reduce_mean(tf.square(residual_T))
    
    # exact loss
    u_pred_e   = model([te_tf, re_tf, Se_tf, kappae_tf, sigma_re_tf, sigma_Se_tf, rhoe_tf])
    residual_e = u_pred_e - He_tf
    L_e        = tf.reduce_mean(tf.square(residual_e))
    L_e        = 0 # HERE delete this to use exact error
    
    # weighted loss and total loss
    weighted_loss = loss_weights[0]*L_D + loss_weights[1]*L_T + loss_weights[2]*L_e
    c = np.zeros_like(loss_weights)
    c[:] = loss_weights
    c[np.where(loss_weights==0)] = 1
    tot_loss_weights = loss_weights/c
    total_loss = tot_loss_weights[0]*L_D + tot_loss_weights[1]*L_T + tot_loss_weights[2]*L_e
    
    return weighted_loss, L_D, L_T, L_e, total_loss

#%% Definition of the training function (used for each neural network)

def train_function(model, t,  rt, St, kappa,  sigma_r,  sigma_S,  rho,
                              rT, ST, kappaT, sigma_rT, sigma_ST, rhoT, HT,
                          te, re, Se, kappae, sigma_re, sigma_Se, rhoe, He,
                          T, sc, ns, loss_weights, epochs=10000, lr=0.001, batch_perc=0.20):
    
    # optimizer choice
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # batch calculation
    num_samples = len(rt)
    batch_size  = int(np.round(batch_perc * num_samples, decimals=0))
    num_batches = int(num_samples // batch_size)  # inner, D
    num_sampleT = len(rT)
    batch_sizeT = num_sampleT // num_batches  # maturity boundary, T
    num_samplee = len(re)
    batch_sizee = num_samplee // num_batches  # exact boundary, e
    
    # creation of a loss table
    losses = np.zeros((epochs, 5), dtype=float)
    
    for epoch in range(epochs):
        # shuffle the data indices for each epoch
        indices  = np.random.permutation(num_samples)
        indicesT = np.random.permutation(num_sampleT)
        indicese = np.random.permutation(num_samplee)
        weight_loss  = 0.0
        total_loss_D = 0.0
        total_loss_T = 0.0
        total_loss_e = 0.0
        total_loss   = 0.0
        
        for batch in range(num_batches):
            # indices in the batch
            batch_indices  = indices[batch * batch_size: (batch + 1) * batch_size]
            batch_indicesT = indicesT[batch * batch_sizeT: (batch + 1) * batch_sizeT]
            batch_indicese = indicese[batch * batch_sizee: (batch + 1) * batch_sizee]
            
            # inputs for L_D
            t_batch       = t[batch_indices]
            rt_batch      = rt[batch_indices]
            St_batch      = St[batch_indices]
            kappa_batch   = kappa[batch_indices]
            sigma_r_batch = sigma_r[batch_indices]
            sigma_S_batch = sigma_S[batch_indices]
            rho_batch     = rho[batch_indices]

            # inputs for L_T
            rT_batch       = rT[batch_indicesT]
            ST_batch       = ST[batch_indicesT]
            kappaT_batch   = kappaT[batch_indicesT]
            sigma_rT_batch = sigma_rT[batch_indicesT]
            sigma_ST_batch = sigma_ST[batch_indicesT]
            rhoT_batch     = rhoT[batch_indicesT]
            HT_batch       = HT[batch_indicesT]
            
            # inputs for L_e
            te_batch       = te[batch_indicese]
            re_batch       = re[batch_indicese]
            Se_batch       = Se[batch_indicese]
            kappae_batch   = kappae[batch_indicese]
            sigma_re_batch = sigma_re[batch_indicese]
            sigma_Se_batch = sigma_Se[batch_indicese]
            rhoe_batch     = rhoe[batch_indicese]
            He_batch       = He[batch_indicese]
            
            # loss calculation
            with tf.GradientTape() as tape:
                # computation of L_weighted, L_D, L_T, L_e, L_tot
                loss, L_D, L_T, L_e, L_tot = loss_function(model, 
                                                           t_batch,  rt_batch, St_batch, kappa_batch,  sigma_r_batch,  sigma_S_batch,  rho_batch,
                                                                     rT_batch, ST_batch, kappaT_batch, sigma_rT_batch, sigma_ST_batch, rhoT_batch, HT_batch,
                                                           te_batch, re_batch, Se_batch, kappae_batch, sigma_re_batch, sigma_Se_batch, rhoe_batch, He_batch,
                                                           T, sc, ns, loss_weights)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            weight_loss   += loss
            total_loss_D  += L_D
            total_loss_T  += L_T
            total_loss_e  += L_e
            total_loss    += L_tot
        avg_loss = weight_loss / num_batches
        losses[epoch, :] = [avg_loss, total_loss_D / num_batches, total_loss_T / num_batches,
                            total_loss_e / num_batches, total_loss / num_batches]
        
        # to see the progress of the calibration
        if (epoch + 1) % 100 == 0:
            print("Epoch {}/{}: Loss = {}".format(epoch + 1, epochs, avg_loss.numpy()))
            
    return losses

#%% Training the neural networks or loading weights : loop on the time steps/neural networks

def on_all_periods(models, t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho,
                                     rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT,
                           te_tilde, re_tilde, Se_tilde, kappae, sigma_re, sigma_Se, rhoe, He,         
                           I, sc, ns, loss_weights, K, train_phases, TRAIN=True):
    
    T = I[-1]
    num_samp    = int(np.shape(rt_tilde)[0]/T)
    num_sampT   = int(np.shape(rT_tilde)[0]/T)
    num_sampe   = int(np.shape(re_tilde)[0]/T)
    id_phase    = np.shape(train_phases)[0]
    epochs      = int(sum(train_phases)[0])
    train_times = np.zeros(T)
    losses_tab  = np.zeros((T,epochs,5), dtype=float)
    
    # previous calibrated neural network for exercise date payoff calculation
    Nv_prev = np.zeros(num_sampT).reshape(-1,1)
    
    # backward training of the neural networks
    for i in range(T-1,-1,-1):
        
        # current neural network
        N = models[i]
        
        T_tilde_i = sc['ah']+sc['bh']*I[i+1]*np.ones(num_sampT).reshape(-1,1)
        
        # inputes for inner loss
        t_tilde_i  = t_tilde[num_samp*i:num_samp*(i+1)]
        rt_tilde_i = rt_tilde[num_samp*i:num_samp*(i+1)]
        St_tilde_i = St_tilde[num_samp*i:num_samp*(i+1)]
        kappa_i    = kappa[num_samp*i:num_samp*(i+1)]
        sigma_r_i  = sigma_r[num_samp*i:num_samp*(i+1)]
        sigma_S_i  = sigma_S[num_samp*i:num_samp*(i+1)]
        rho_i      = rho[num_samp*i:num_samp*(i+1)]
        
        # inputs for maturity loss
        rT_tilde_i = rT_tilde[num_sampT*i:num_sampT*(i+1)]
        ST_tilde_i = ST_tilde[num_sampT*i:num_sampT*(i+1)]
        kappaT_i   = kappaT[num_sampT*i:num_sampT*(i+1)]
        sigma_rT_i = sigma_rT[num_sampT*i:num_sampT*(i+1)]
        sigma_ST_i = sigma_ST[num_sampT*i:num_sampT*(i+1)]
        rhoT_i     = rhoT[num_sampT*i:num_sampT*(i+1)]
        if I[i+1]==T:
            HT_i = np.maximum(K-(ST_tilde_i-sc["aS"])/sc["bS"],np.zeros_like(K-(ST_tilde_i-sc["aS"])/sc["bS"]))
        else:
            N_temp  = models[T-i-2]
            Nv_prev = N_temp([T_tilde_i, rT_tilde_i, ST_tilde_i, kappaT_i, sigma_rT_i, sigma_ST_i, rhoT_i])
            HT_i    = np.max([np.maximum(K-(ST_tilde_i-sc["aS"])/sc["bS"],np.zeros_like(K-(ST_tilde_i-sc["aS"])/sc["bS"])),Nv_prev],axis=0)
        
        #inputs for exact loss
        te_tilde_i = te_tilde[num_sampe*i:num_sampe*(i+1)]
        re_tilde_i = re_tilde[num_sampe*i:num_sampe*(i+1)]
        Se_tilde_i = Se_tilde[num_sampe*i:num_sampe*(i+1)]
        kappae_i   = kappae[num_sampe*i:num_sampe*(i+1)]
        sigma_re_i = sigma_re[num_sampe*i:num_sampe*(i+1)]
        sigma_Se_i = sigma_Se[num_sampe*i:num_sampe*(i+1)]
        rhoe_i     = rhoe[num_sampe*i:num_sampe*(i+1)]
        He_i       = He[num_sampe*i:num_sampe*(i+1)]
        
        if TRAIN:
            tic = timeit.default_timer()
            for tr in range(id_phase):
                n_tot_epochs = int(0)
                for ep in range(tr):
                    n_tot_epochs += int(train_phases[ep,0])
                n_epochs = int(train_phases[tr,0])
                losses_tab[i,n_tot_epochs:n_tot_epochs+n_epochs,:] = train_function(N, t_tilde_i,  rt_tilde_i, St_tilde_i, kappa_i,  sigma_r_i,  sigma_S_i,  rho_i, 
                                                                                                   rT_tilde_i, ST_tilde_i, kappaT_i, sigma_rT_i, sigma_ST_i, rhoT_i, HT_i, 
                                                                                       te_tilde_i, re_tilde_i, Se_tilde_i, kappae_i, sigma_re_i, sigma_Se_i, rhoe_i, He_i, 
                                                                                       sc['ah']+sc['bh']*I[i+1], sc, ns, loss_weights, 
                                                                                       epochs=n_epochs, lr=train_phases[tr,1], batch_perc=0.34)
            toc = timeit.default_timer()
            train_times[i] = toc-tic
            print('Training time for the neural network {} is of {} sec.'.format(i,round(train_times[i],1)))
            
            N.count_params()
            N.save_weights(Directory_name +"/" + weight_name + "__" + str(i) + ".h5")   
            if i==T-1:
                np.save(Directory_name +"/" + scaling_ay_name,[sc["ar"],sc["aS"]])
                np.save(Directory_name +"/" + scaling_by_name,[sc["br"],sc["bS"]])
            
            loss_tab = losses_tab[i,:,:].reshape((epochs,5))
            print("Weighted Loss:", loss_tab[:,0][-1])
            print("Inner Loss:", loss_tab[:,1][-1])
            print("Maturity Loss:", loss_tab[:,2][-1])
            print("Exact Loss:", loss_tab[:,3][-1])
            print("Total Loss:", loss_tab[:,4][-1])
            
            fig, ax = plt.subplots(2, 2)          
            plt.subplot(2, 2, 1)
            ax[0, 0].plot(np.arange(0,epochs), loss_tab[:,0])
            plt.subplot(2, 2, 1)
            ax[0, 1].plot(np.arange(0,epochs), loss_tab[:,1])
            plt.subplot(2, 2, 1)
            ax[1, 0].plot(np.arange(0,epochs), loss_tab[:,2])
            plt.subplot(2, 2, 1)
            ax[1, 1].plot(np.arange(0,epochs), loss_tab[:,3])

            ax[0, 0].set_title("Weighted Loss")
            ax[0, 1].set_title("Inner Loss")
            ax[1, 0].set_title("Maturity Loss")
            ax[1, 1].set_title("Exact Loss")

            fig.tight_layout()
            fig.savefig(Directory_name +"/" + plots_loss_name + "__" + str(i) + ".png")
            plt.show()

        else:
            # initializing the neural network (tmp not used after)
            tmp = N([t_tilde_i, rt_tilde_i, St_tilde_i, kappa_i, sigma_r_i, sigma_S_i, rho_i])
            # loading the calibrated weights
            N.load_weights(Directory_name +"/" + weight_name + "__" + str(i) + ".h5")
            if i==T-1:
                # loading the scaling parameters (same for all networks)
                ay = np.load(Directory_name +"/" + scaling_ay_name)
                by = np.load(Directory_name +"/" + scaling_by_name)

            # running only 10 epochs for getting the losses
            losses_tab[i,:10,:] = train_function(N, t_tilde_i,  rt_tilde_i, St_tilde_i, kappa_i,  sigma_r_i,  sigma_S_i,  rho_i, 
                                                                rT_tilde_i, ST_tilde_i, kappaT_i, sigma_rT_i, sigma_ST_i, rhoT_i, HT_i, 
                                                    te_tilde_i, re_tilde_i, Se_tilde_i, kappae_i, sigma_re_i, sigma_Se_i, rhoe_i, He_i, 
                                                    sc['ah']+sc['bh']*I[i+1], sc, ns, loss_weights, 
                                                    epochs=10, lr=train_phases[-1,1], batch_perc=0.34)
            losses_tab = losses_tab[:,:10,:]
        
    if TRAIN:
        print("Total training time is of {} sec.".format(sum(train_times)))
        return models, train_times, losses_tab
    else:
        return models, ay, by, losses_tab

#%% Main

#%% Definition of parameters and range values

# Nelson-Siegel parameters
b0  =  0.03974889670677964
b10 =  0.0014721430763244212
b11 = -0.01693616284948873
c1  =  0.4222897254760573

# Lower and upper values of parameters and state varaibles
r_l       =  0.005; r_u       = 0.07  # stochastic interest rate
S_l       =  20;    S_u       = 250   # stock value
kappa_l   =  0.5;   kappa_u   = 2     # mean-reverting phenomenon of rt
sigma_r_l =  0.002; sigma_r_u = 0.02  # standard deviation of rt
sigma_S_l =  0.02;  sigma_S_u = 0.3   # standard deviation of St
rho_l     = -0.8;   rho_u     = 0.8   # correlation between rt and St

# Fixed parameters
T = 5                                 # maturity of the option
n = 5                                 # number of exercise dates
I = np.linspace(0, T, n+1, dtype=int) # serie of exercise date + initial date (t=0)
K = 100                               # strike price of the option

# Hyperparameters of the neural network
NN_type = "RNN"                       # architecture of the neural network: 
                                          # FNN (feedforward neural network), 
                                          # RNN (residual neural network = FNN with skip connections) or 
                                          # DGM (neural network with Deep Galerkin Method layers)
num_hidden_layers = 3                 # number of hidden layers in the network
num_neurons = 256                     # number of neurons by hidden layer

# Sample sets size for training the neural network
num_samples = 20000                   # number of samples for the inner loss
num_samples_bound = 5000              # number of samples for the maturity loss
num_samples_exact = 400 # HERE        # number of samples for the exact loss
# Sample set size for vizualizing the result
num_samples_test = 50

# Loss weights in the total loss
inner_loss_weight    = 1
maturity_loss_weight = 1
exact_loss_weight    = 0 # HERE
loss_weights = np.array([inner_loss_weight,maturity_loss_weight,exact_loss_weight])

# Learning rates and epochs
lr_1 = 0.005;  epochs_1 = 500
lr_2 = 0.002;  epochs_2 = 1000
lr_3 = 0.001;  epochs_3 = 1000
lr_4 = 0.0001; epochs_4 = 1000
train_phases = np.array([[epochs_1,lr_1],[epochs_2,lr_2],[epochs_3,lr_3],[epochs_4,lr_4]])

# Directory and file names for saving weights, scaling parameters and results
Directory_name = "Berm_one_gamma__"+str(T)+"_"+str(n)+"__"+NN_type+"_"+str(num_hidden_layers)+"_"+str(num_neurons)+"__"+str(num_samples)+"_"+str(num_samples_bound)+"_"+str(num_samples_exact)+"__"+str(inner_loss_weight)+"_"+str(maturity_loss_weight)+"_"+str(exact_loss_weight)
exists = os.path.exists(Directory_name)
if not exists:
    os.mkdir(Directory_name)
weight_name     = "weights__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
scaling_ay_name = "scaling_ay__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".npy"
scaling_by_name = "scaling_by__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)+".npy"
plots_loss_name = "plots_loss__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
plot_singleset_rt_name = "plot_single_set_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
plot_singleset_St_name = "plot_single_set_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
heatmap_alea_rt_name = "heatmap_alea_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
heatmap_alea_St_name = "heatmap_alea_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
heatmap_conf_rt_name = "heatmap_conf_rt__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)
heatmap_conf_St_name = "heatmap_conf_St__"+str(lr_1)+"_"+str(epochs_1)+"_"+str(lr_2)+"_"+str(epochs_2)+"_"+str(lr_3)+"_"+str(epochs_3)+"_"+str(lr_4)+"_"+str(epochs_4)

#%% Construction of sample sets for training

# for reproductability
np.random.seed(23)

# Inner sample set
tot_num_samp = T*num_samples
t = np.zeros(tot_num_samp).reshape(-1, 1)
for i in range(T):
    t[num_samples*i:num_samples*(i+1)] = np.random.uniform(I[i], I[i+1], num_samples).reshape(-1, 1)
rt      = np.random.uniform(r_l, r_u, tot_num_samp).reshape(-1, 1)
St      = np.random.uniform(S_l, S_u, tot_num_samp).reshape(-1, 1)
kappa   = np.random.uniform(kappa_l, kappa_u, tot_num_samp).reshape(-1, 1)
sigma_r = np.random.uniform(sigma_r_l, sigma_r_u, tot_num_samp).reshape(-1, 1)
sigma_S = np.random.uniform(sigma_S_l, sigma_S_u, tot_num_samp).reshape(-1, 1)
rho     = np.random.uniform(rho_l, rho_u, tot_num_samp).reshape(-1, 1)
# scaling of state variables
rbar = np.mean(rt); Sr = np.std(rt)
Sbar = np.mean(St); SS = np.std(St)
ar = -rbar/Sr; br = 1/Sr 
aS = -Sbar/SS; bS = 1/SS
ah = -1/2;     bh = 1/T

rt_tilde = ar+br*rt
St_tilde = aS+bS*St
t_tilde  = ah+bh*t

# Maturity sample set
tot_num_sampT = T*num_samples_bound
rT       = np.random.uniform(r_l, r_u, tot_num_sampT).reshape(-1, 1)
ST       = np.random.uniform(S_l, S_u, tot_num_sampT).reshape(-1, 1)
kappaT   = np.random.uniform(kappa_l, kappa_u, tot_num_sampT).reshape(-1, 1)
sigma_rT = np.random.uniform(sigma_r_l, sigma_r_u, tot_num_sampT).reshape(-1, 1)
sigma_ST = np.random.uniform(sigma_S_l, sigma_S_u, tot_num_sampT).reshape(-1, 1)
rhoT     = np.random.uniform(rho_l, rho_u, tot_num_sampT).reshape(-1, 1)
# scaling of state variables
rT_tilde = ar+br*rT
ST_tilde = aS+bS*ST

# Exact sample set
tot_num_sampe = T*num_samples_exact
te = np.zeros(tot_num_sampe).reshape(-1, 1)
for i in range(T):
    te[num_samples_exact*i:num_samples_exact*(i+1)] = np.random.uniform(i, i+1, num_samples_exact).reshape(-1, 1)
re       = np.random.uniform(r_l, r_u, tot_num_sampe).reshape(-1, 1)
Se       = np.random.uniform(S_l, S_u, tot_num_sampe).reshape(-1, 1)
kappae   = np.random.uniform(kappa_l, kappa_u, tot_num_sampe).reshape(-1, 1)
sigma_re = np.random.uniform(sigma_r_l, sigma_r_u, tot_num_sampe).reshape(-1, 1)
sigma_Se = np.random.uniform(sigma_S_l, sigma_S_u, tot_num_sampe).reshape(-1, 1)
rhoe     = np.random.uniform(rho_l, rho_u, tot_num_sampe).reshape(-1, 1)
# scaling of state variables
re_tilde = ar+br*re
Se_tilde = aS+bS*Se
te_tilde = ah+bh*te
# exact value of the option
He = np.zeros(tot_num_sampe).reshape(-1, 1) # HERE

#%% Initialization of a new model

# dictionnary for scaling parameters and Nelson-Siegel parameters
sc = {'ar':ar, 'br':br, 'aS':aS, 'bS':bS, 'ah':ah, 'bh':bh}
ns = {'b0':b0, 'b10':b10, 'b11':b11, 'c1':c1}

models = np.array([])
for i in range(T):
    if NN_type=="FNN":
        models = np.append(models,FNN(num_neurons, num_hidden_layers))
    elif NN_type=="RNN":
        models = np.append(models,RNN(num_neurons, num_hidden_layers))
    elif NN_type=="DGM":
        models = np.append(models,DGM(num_neurons, num_hidden_layers))
        
#%% Training of a new model (TRAIN=True) or loading of weights (TRAIN=False)

TRAIN = False

if TRAIN:
    models_out, train_times, losses_table = on_all_periods(models, t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho, 
                                                                             rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, 
                                                                   te_tilde, re_tilde, Se_tilde, kappae, sigma_re, sigma_Se, rhoe, He, 
                                                                   I, sc, ns, loss_weights, K, train_phases, TRAIN=True)
else:
    models_out, ay, by, losses_table = on_all_periods(models, t_tilde,  rt_tilde, St_tilde, kappa,  sigma_r,  sigma_S,  rho, 
                                                                        rT_tilde, ST_tilde, kappaT, sigma_rT, sigma_ST, rhoT, 
                                                              te_tilde, re_tilde, Se_tilde, kappae, sigma_re, sigma_Se, rhoe, He, 
                                                              I, sc, ns, loss_weights, K, train_phases, TRAIN=False)
    train_times = np.zeros(n)
    
#%% Validation: comparison with another pricing method
"""
This alternative pricing method is used :
    - to provide a set of exact prices for training the model
    - to validate the results obtained with neural networks.
Unfortunately, there is no efficient pricing method available at the moment. 
A tree-based method has been developed using a hybrid tree for rt (trinomial tree) and St (binomial tree). 
But it cannot be used in practice because the number of states grows exponentially and it is impossible to free up enough memory to take more than 9 time steps. 
However, such a method is only accurate for a number of time steps tending towards infinity.

This code is designed to work perfectly with the other pricing method (see the lines with comment "# HERE" to modify). 
In the meantime, we do not consider the loss on the exact prices in the model training (exact_loss_weight = 0) and 
it is impossible for us to compare our results and obtain measures of error (MSE or relative error). 
We just display the results of the neural networks in order to comment them.
"""
#%% u_pred vs. rt with a single set of parameters

# state variables and parameters
rt_test      = np.linspace(0.005, 0.07, num_samples_test).reshape(-1, 1)
St_test      = np.zeros_like(rt_test).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
St_test[:]      =  115
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# loop on the models
for i in range(n):
    t_test[:]     = I[i]
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[i]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # fair value given by another pricing method
    F_op = np.zeros(num_samples_test) # HERE
    
    # plot of PINN & other pricing method prices
    #plt.plot(rt_test,F_op, "-g", label="u_other")
    plt.plot(rt_test, F_NN, "-b", label = "PINN price of network N{} at t = {}".format(i+1,t_test[0][0]))
    #plt.plot(rt_test, F_NN - F_op, "-r", label = "difference")
    plt.legend()
    plt.xlabel('$r_t$')
    plt.ylabel('$u$')
    plt.savefig(Directory_name +"/" + plot_singleset_rt_name + "__" + str(i) + ".png", bbox_inches='tight')
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
K_test       = np.zeros_like(St_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# loop on the models
for i in range(n):
    t_test[:]     = I[i]
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[i]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # fair value given by another pricing method
    F_op = np.zeros(num_samples_test) # HERE
    
    # plot of PINN & other pricing method prices
    plt.plot(St_test, np.maximum(K_test-St_test,np.zeros_like(St_test)), "--k", label="Payoff")
    #plt.plot(St_test,F_op, "-g", label="u_other")
    plt.plot(St_test, F_NN, "-b", label = "PINN price of network N{} at t = {}".format(i+1,t_test[0][0]))
    #plt.plot(St_test, F_NN - F_op, "-r", label = "difference")
    plt.legend()
    plt.xlabel('$S_t$')
    plt.ylabel('$u$')
    plt.savefig(Directory_name +"/" + plot_singleset_St_name + "__" + str(i) + ".png", bbox_inches='tight')
    plt.show()

#%% u_pred vs. St with a single set of parameters

# state variables and parameters
# St_test      = np.linspace(70, 130, num_samples_test).reshape(-1, 1)
St_test      = np.linspace(95, 105, num_samples_test).reshape(-1, 1)
rt_test      = np.zeros_like(St_test).reshape(-1, 1)
kappa_test   = np.zeros_like(St_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(St_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(St_test).reshape(-1, 1)
rho_test     = np.zeros_like(St_test).reshape(-1, 1)
t_test       = np.zeros_like(St_test).reshape(-1, 1)
K_test       = np.zeros_like(St_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

plt.figure()
plt.plot(St_test, np.maximum(K_test-St_test,np.zeros_like(St_test)), "--k", label="Payoff")

# loop on the models
for i in range(n):
    t_test[:]     = I[i+1]
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[i]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # plot of PINN price
    plt.plot(St_test, F_NN, label = "PINN price of network N{} at t = {}".format(i+1,int(t_test[0][0])))
plt.legend()
plt.xlabel('$S_t$')
plt.ylabel('$u$')
plt.savefig(Directory_name +"/" + plot_singleset_St_name + ".png", bbox_inches='tight')
plt.show()

#%% fixed rt and t, rest random

np.random.seed(24)

# random sample test
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.random.uniform(S_l, S_u, num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_alea_rt = np.zeros(n)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], num_samples_test))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN       = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
    # fair value given by another pricing method
    F_op       = np.zeros(num_samples_test**2) # HERE
    F_op       = F_op.reshape(-1, 1)
    F_op_array = np.array(F_op).reshape(num_samples_test, num_samples_test)
    
    # difference
    diff       = np.abs(F_NN - F_op)
    diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)
    
    # plot of PINN & other pricing method prices: heatmap rt/t
    df = pd.DataFrame(F_NN_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    #df = pd.DataFrame(diff_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('$r_t$ [%]')
    plt.ylabel('$t$ [year]')
    plt.savefig(Directory_name +"/" + heatmap_alea_rt_name + "__" + str(k) + ".png", bbox_inches='tight')
    plt.show()
    
    MSE_alea_rt_k  = F_NN - F_op
    MSE_alea_rt[k] = np.mean(np.square(MSE_alea_rt_k[~np.isnan(MSE_alea_rt_k)]))
    
total_MSE_alea_rt = np.sum(MSE_alea_rt)
print("Mean Square Errors:", MSE_alea_rt, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_alea_rt, "WRONG, no validation model")

#%% fixed rt and t, rest random

np.random.seed(24)

# random sample test
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.random.uniform(S_l, S_u, num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

F_NN = np.zeros_like(rt_test).reshape(-1, 1)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], int(num_samples_test/n)))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(int(num_samples_test))])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN[500*k:500*(k+1)] = models_out[k]([t_tilde_test, rt_tilde_test[500*k:500*(k+1)], St_tilde_test[500*k:500*(k+1)], kappa_test[500*k:500*(k+1)], sigma_r_test[500*k:500*(k+1)], sigma_S_test[500*k:500*(k+1)], rho_test[500*k:500*(k+1)]])

F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
# plot of PINN price: heatmap rt/t
df = pd.DataFrame(F_NN_array, index = [T*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$r_t$ [%]')
plt.ylabel('$t$ [year]')
plt.savefig(Directory_name +"/" + heatmap_alea_rt_name + ".png", bbox_inches='tight')
plt.show()

#%% fixed St and t, rest random

np.random.seed(25)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, num_samples_test**2).reshape(-1, 1)
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_alea_St = np.zeros(n)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], num_samples_test))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN       = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
    # fair value given by another pricing method
    F_op       = np.zeros(num_samples_test**2) # HERE
    F_op       = F_op.reshape(-1, 1)
    F_op_array = np.array(F_op).reshape(num_samples_test, num_samples_test)
    
    # difference
    diff       = np.abs(F_NN - F_op)
    diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)
    
    # plot of PINN & other pricing method prices: heatmap St/t
    df = pd.DataFrame(F_NN_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    #df = pd.DataFrame(diff_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('$S_t$')
    plt.ylabel('$t$ [year]')
    plt.savefig(Directory_name +"/" + heatmap_alea_St_name + "__" + str(k) + ".png", bbox_inches='tight')
    plt.show()
    
    MSE_alea_St_k  = F_NN - F_op
    MSE_alea_St[k] = np.mean(np.square(MSE_alea_St_k[~np.isnan(MSE_alea_St_k)]))
    
total_MSE_alea_St = np.sum(MSE_alea_St)
print("Mean Square Errors:", MSE_alea_St, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_alea_St, "WRONG, no validation model")

#%% fixed St and t, rest random

np.random.seed(25)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, num_samples_test**2).reshape(-1, 1)
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, num_samples_test**2).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

F_NN = np.zeros_like(rt_test).reshape(-1, 1)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], int(num_samples_test/n)))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN[500*k:500*(k+1)] = models_out[k]([t_tilde_test, rt_tilde_test[500*k:500*(k+1)], St_tilde_test[500*k:500*(k+1)], kappa_test[500*k:500*(k+1)], sigma_r_test[500*k:500*(k+1)], sigma_S_test[500*k:500*(k+1)], rho_test[500*k:500*(k+1)]])
    
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
# plot of PINN price: heatmap St/t
df = pd.DataFrame(F_NN_array, index = [T*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$S_t$')
plt.ylabel('$t$ [year]')
plt.savefig(Directory_name +"/" + heatmap_alea_St_name + ".png", bbox_inches='tight')
plt.show()

#%% fixed rt and t, rest in a particular configuration

# state variables and parameters
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.zeros_like(rt_test).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
St_test[:]      =  90
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_conf_rt = np.zeros(n)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], num_samples_test))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN       = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
    # fair value given by another pricing method
    F_op       = np.zeros(num_samples_test**2) # HERE
    F_op       = F_op.reshape(-1, 1)
    F_op_array = np.array(F_op).reshape(num_samples_test, num_samples_test)
    
    # difference
    diff       = np.abs(F_NN - F_op)
    diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)
    
    # plot of PINN & other pricing method prices: heatmap rt/t
    df = pd.DataFrame(F_NN_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    #df = pd.DataFrame(diff_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('$r_t$ [%]')
    plt.ylabel('$t$ [year]')
    plt.savefig(Directory_name +"/" + heatmap_conf_rt_name + "__" + str(k) + ".png", bbox_inches='tight')
    plt.show()
    
    MSE_conf_rt_k  = F_NN - F_op
    MSE_conf_rt[k] = np.mean(np.square(MSE_conf_rt_k[~np.isnan(MSE_conf_rt_k)]))
    
total_MSE_conf_rt = np.sum(MSE_conf_rt)
print("Mean Square Errors:", MSE_conf_rt, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_conf_rt, "WRONG, no validation model")

#%% fixed rt and t, rest in a particular configuration

# state variables and parameters
rt_test      = np.array([np.linspace(r_l, r_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
St_test      = np.zeros_like(rt_test).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
St_test[:]      =  90
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

F_NN = np.zeros_like(rt_test).reshape(-1, 1)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], int(num_samples_test/n)))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN[500*k:500*(k+1)] = models_out[k]([t_tilde_test, rt_tilde_test[500*k:500*(k+1)], St_tilde_test[500*k:500*(k+1)], kappa_test[500*k:500*(k+1)], sigma_r_test[500*k:500*(k+1)], sigma_S_test[500*k:500*(k+1)], rho_test[500*k:500*(k+1)]])
    
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
# F_NN_array = F_NN_array[10:,:]
    
# plot of PINN price: heatmap rt/t
df = pd.DataFrame(F_NN_array, index = [T*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
# df = pd.DataFrame(F_NN_array, index = [round(1 + (5 - 1)*(i-1)/40, 1) for i in range(1,40 + 1)], columns = [round(r_l*100 + (r_u*100 - r_l*100)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$r_t$ [%]')
plt.ylabel('$t$ [year]')
plt.savefig(Directory_name +"/" + heatmap_conf_rt_name + ".png", bbox_inches='tight')
plt.show()

#%% fixed St and t, rest in a particular configuration

# state variables and parameters
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
rt_test      = np.zeros_like(St_test).reshape(-1, 1)
kappa_test   = np.zeros_like(St_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(St_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(St_test).reshape(-1, 1)
rho_test     = np.zeros_like(St_test).reshape(-1, 1)
t_test       = np.zeros_like(St_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_conf_St = np.zeros(n)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], num_samples_test))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN       = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
    # fair value given by another pricing method
    F_op       = np.zeros(num_samples_test**2) # HERE
    F_op       = F_op.reshape(-1, 1)
    F_op_array = np.array(F_op).reshape(num_samples_test, num_samples_test)
    
    # difference
    diff       = np.abs(F_NN - F_op)
    diff_array = np.array(diff).reshape(num_samples_test,num_samples_test)
    
    # plot of PINN & other pricing method prices: heatmap St/t
    df = pd.DataFrame(F_NN_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    #df = pd.DataFrame(diff_array, index = [round(I[k] + (I[k+1] - I[k])*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
    sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('$S_t$')
    plt.ylabel('$t$ [year]')
    plt.savefig(Directory_name +"/" + heatmap_conf_St_name + "__" + str(k) + ".png", bbox_inches='tight')
    plt.show()
    
    MSE_conf_St_k  = F_NN - F_op
    MSE_conf_St[k] = np.mean(np.square(MSE_conf_St_k[~np.isnan(MSE_conf_St_k)]))
    
total_MSE_conf_St = np.sum(MSE_conf_St)
print("Mean Square Errors:", MSE_conf_St, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_conf_St, "WRONG, no validation model")

#%% fixed St and t, rest in a particular configuration

# state variables and parameters
St_test      = np.array([np.linspace(S_l, S_u, num_samples_test) for i in range(num_samples_test)]).reshape(-1,1)
rt_test      = np.zeros_like(St_test).reshape(-1, 1)
kappa_test   = np.zeros_like(St_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(St_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(St_test).reshape(-1, 1)
rho_test     = np.zeros_like(St_test).reshape(-1, 1)
t_test       = np.zeros_like(St_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
rt_test[:]      =  0.03
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

F_NN = np.zeros_like(rt_test).reshape(-1, 1)

# loop on the models
for k in range(n):
    a = list(np.linspace(I[k], I[k+1], int(num_samples_test/n)))
    t_test = []
    for i in range(len(a)):
        t_test.append([a[i] for j in range(num_samples_test)])
    t_test = np.array(t_test).flatten().reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN[500*k:500*(k+1)] = models_out[k]([t_tilde_test, rt_tilde_test[500*k:500*(k+1)], St_tilde_test[500*k:500*(k+1)], kappa_test[500*k:500*(k+1)], sigma_r_test[500*k:500*(k+1)], sigma_S_test[500*k:500*(k+1)], rho_test[500*k:500*(k+1)]])
    
F_NN_array = np.array(F_NN).reshape(num_samples_test,num_samples_test)
    
# plot of PINN price: heatmap St/t
df = pd.DataFrame(F_NN_array, index = [T*i/num_samples_test for i in range(0,num_samples_test)], columns = [round(S_l + (S_u - S_l)*(i-1)/num_samples_test, 1) for i in range(1,num_samples_test + 1)])
sns.heatmap(df, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('$S_t$')
plt.ylabel('$t$ [year]')
plt.savefig(Directory_name +"/" + heatmap_conf_St_name + ".png", bbox_inches='tight')
plt.show()

#%%  all random

np.random.seed(26)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(S_l, S_u, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, 2*num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, 2*num_samples_test**2).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_alea = np.zeros(n)

# loop on the models
for k in range(n):
    for i in range(2*num_samples_test**2):
        t_test[i]  = np.random.uniform(I[k], I[k+1], 1).reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # fair value given by another pricing method
    F_op = np.zeros(2*num_samples_test**2) # HERE
    F_op = F_op.reshape(-1, 1)
    
    MSE_alea_k  = F_NN - F_op
    MSE_alea[k] = np.mean(np.square(MSE_alea_k[~np.isnan(MSE_alea_k)]))
    
total_MSE_alea = np.sum(MSE_alea)
print("Mean Square Errors:", MSE_alea, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_alea, "WRONG, no validation model")

#%% state variables rt and St random, rest in a particular configuration

np.random.seed(27)
# random sample test for state variables
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(S_l, S_u, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.zeros_like(rt_test).reshape(-1, 1)
sigma_r_test = np.zeros_like(rt_test).reshape(-1, 1)
sigma_S_test = np.zeros_like(rt_test).reshape(-1, 1)
rho_test     = np.zeros_like(rt_test).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
# values
kappa_test[:]   =  1.15
sigma_r_test[:] =  0.01 
sigma_S_test[:] =  0.025
rho_test[:]     = -0.4
K_test[:]       =  K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for MSE calculations
MSE_conf = np.zeros(n)

# loop on the models
for k in range(n):
    t_test[:]     = I[k]
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # fair value given by another pricing method
    F_op = np.zeros(2*num_samples_test**2) # HERE
    F_op = F_op.reshape(-1, 1)
    
    MSE_conf_k  = F_NN - F_op
    MSE_conf[k] = np.mean(np.square(MSE_conf_k[~np.isnan(MSE_conf_k)]))
    
total_MSE_conf = np.sum(MSE_conf)
print("Mean Square Errors:", MSE_conf, "WRONG, no validation model")
print("Total Mean Square Error:", total_MSE_conf, "WRONG, no validation model")

#%% relative error in the money (St<K), all random

np.random.seed(28)
# random sample test
rt_test      = np.random.uniform(r_l, r_u, 2*num_samples_test**2).reshape(-1, 1)
St_test      = np.random.uniform(S_l, K, 2*num_samples_test**2).reshape(-1, 1)
kappa_test   = np.random.uniform(kappa_l, kappa_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_r_test = np.random.uniform(sigma_r_l, sigma_r_u, 2*num_samples_test**2).reshape(-1, 1)
sigma_S_test = np.random.uniform(sigma_S_l, sigma_S_u, 2*num_samples_test**2).reshape(-1, 1)
rho_test     = np.random.uniform(rho_l, rho_u, 2*num_samples_test**2).reshape(-1, 1)
t_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test       = np.zeros_like(rt_test).reshape(-1, 1)
K_test[:]    = K
# scaling of state variables
rt_tilde_test = ar+br*rt_test
St_tilde_test = aS+bS*St_test

# array for error calculations
err_relative = np.zeros(n)

# loop on the models
for k in range(n):
    for i in range(2*num_samples_test**2):
        t_test[i]  = np.random.uniform(I[k], I[k+1], 1).reshape(-1, 1)
    t_tilde_test  = ah+bh*t_test
    
    # fair value given by the neural network
    F_NN = models_out[k]([t_tilde_test, rt_tilde_test, St_tilde_test, kappa_test, sigma_r_test, sigma_S_test, rho_test])
    
    # fair value given by another pricing method
    F_op = np.ones(2*num_samples_test**2) # HERE
    F_op = F_op.reshape(-1, 1)
    
    err  = np.abs(F_NN - F_op)
    err  = err[~np.isnan(F_op)]
    F_op = F_op[~np.isnan(F_op)]
    err  = err[np.nonzero(F_op)]
    F_op = F_op[np.nonzero(F_op)]
    err_relative[k] = np.mean(err/np.abs(F_op))
    
total_err_relative = np.sum(err_relative)
print("Relative error in the money:", err_relative, "WRONG, no validation model")
print("Total relative error in the money:", total_err_relative, "WRONG, no validation model")
   
#%% Saving outputs in a csv file

MSE      = [total_MSE_alea_rt, total_MSE_alea_St, total_MSE_conf_rt, total_MSE_conf_St, 
            total_MSE_alea, total_MSE_conf,  total_err_relative]

loss_tab = np.sum(losses_table, axis = 0)

exists_resloc = os.path.exists(Directory_name +"/" +'results.csv')
if not exists_resloc:
    with open(Directory_name +"/" +'results.csv', 'w', newline='') as file:
          writer = csv.writer(file, delimiter=';')
          writer.writerow(["Configuration","Weighted Loss", "Inner Loss", "Exercise Loss", "Lower Loss", "Total Loss",
                          "MSE_alea_rt", "MSE_alea_St", "MSE_conf_rt", "MSE_conf_St", "MSE_alea", 
                          "MSE_conf",  "relative error (in the money)", "Training time [sec]", "TRAIN"])
          writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(sum(train_times)), TRAIN])
else:
    with open(Directory_name +"/" +'results.csv', 'a', newline='') as file:
          writer = csv.writer(file, delimiter=';')
          writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(sum(train_times)), TRAIN])
         
#%% Saving outputs in a csv file countaning outputs of all neural networks

exists_resall = os.path.exists('Berm_results_all.csv')
if not exists_resall:
    with open('Berm_results_all.csv', 'w', newline='') as file:
          writer = csv.writer(file, delimiter=';')
          writer.writerow(["Configuration","Weighted Loss", "Inner Loss", "Exercise Loss", "Lower Loss", "Total Loss",
                          "MSE_alea_rt", "MSE_alea_St", "MSE_conf_rt", "MSE_conf_St", "MSE_alea", 
                          "MSE_conf",  "relative error (in the money)", "Training time [sec]", "TRAIN"])
          writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(sum(train_times)), TRAIN])
else:
    with open('Berm_results_all.csv', 'a', newline='') as file:
          writer = csv.writer(file, delimiter=';')
          writer.writerow([Directory_name, round(loss_tab[:,0][-1], 6) , round(loss_tab[:,1][-1], 6),
                          round(loss_tab[:,2][-1], 6), round(loss_tab[:,3][-1], 6), round(loss_tab[:,4][-1], 6), round(MSE[0], 6),  
                          round(MSE[1], 6), round(MSE[2], 6), round(MSE[3], 6), round(MSE[4], 6), 
                          round(MSE[5], 6), round(MSE[6], 6), int(sum(train_times)), TRAIN])
