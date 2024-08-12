# -*- coding: utf-8 -*-
"""
Validation pricing method for Bermudan option: hybrid tree for rt (trinomial tree) and St (binomial tree)

@author: Marie Thibeau
"""

import numpy as np
from scipy.sparse import csr_matrix

#%% Some useful functions from the Nelson-Siegel model

def f_function(t,b0,b10,b11,c1):
    # forward rate, f(0,t)
    # modelised by the Nelson-Siegel model
    f = b0 + (b10 + b11*t)*np.exp(-c1*t)
    return f

def r0(t,b0,b10,b11,c1):
    # initial zero-coupon yields
    # modelised by the Nelson-Siegel model
    ans = np.zeros_like(t)
    ans[t != 0] = b0 + b10 / (c1 * t[t != 0]) * (1 - np.exp(-c1 * t[t != 0])) + b11 / (t[t != 0] * c1**2) * (1 - (c1 * t[t != 0] + 1) * np.exp(-c1 * t[t != 0]))
    ans[t == 0] = b0
    return ans

#%% Construction of the trinomial tree for the interest rate

def r_tree(T, nt, kap, sg_r, b0, b10, b11, c1):
    dt = T / nt
    t  = np.linspace(0, T, (nt+1))
    V  = np.sqrt(sg_r**2 / (2*kap) * (1 - np.exp(-kap * dt)))
    dx = V * np.sqrt(3)
    
    xmax = 3 * (sg_r / np.sqrt(kap))
    nx = 2 * int(np.ceil(xmax / dx))
    x = nx / 2 * dx - np.arange(nx+1) * dx
    idx_m = nx // 2
    xval = np.tile(x, (nt+1, 1)).T
    
    k  = np.zeros((nx+1, nt+1), dtype=int)
    ku = np.zeros((nx+1, nt+1), dtype=int)
    kd = np.zeros((nx+1, nt+1), dtype=int)
    pm = np.zeros((nx+1, nt+1))
    pu = np.zeros((nx+1, nt+1))
    pd = np.zeros((nx+1, nt+1))
    
    for i in range(nt+1):
        Mi = xval[:, i] * np.exp(-kap * dt)
        k[:, i] = idx_m - np.round(Mi / dx).astype(int)
        eta = Mi - xval[k[:, i], i]
        pu[:, i] = 1/6 + eta**2 / (6 * V**2) + eta / (2 * np.sqrt(3) * V)
        pm[:, i] = 2/3 - eta**2 / (3 * V**2)
        pd[:, i] = 1/6 + eta**2 / (6 * V**2) - eta / (2 * np.sqrt(3) * V)
        
        idx_d = (k[:, i] == 0)
        pm[idx_d, i] += pu[idx_d, i]
        pu[idx_d, i] = 0
        idx_u = (k[:, i] == nx)
        pm[idx_u, i] += pd[idx_u, i]
        pd[idx_u, i] = 0
    
    ZC_yield = r0(np.append(t, max(t) + dt), b0, b10, b11, c1)
    P = np.exp(-ZC_yield * np.append(t, max(t) + dt))
    nu = np.zeros(len(t))
    for i in range(1, nt+2):
        h = np.ones(nx+1)
        for j in range(i - 1, -1, -1):
            r = xval[:, j] + nu[j]
            ku[:, j] = np.clip(k[:, j] - 1, 0, nx)
            kd[:, j] = np.clip(k[:, j] + 1, 0, nx)
            h = np.exp(-r * dt) * (h[ku[:, j]] * pu[:, j] + h[k[:, j]] * pm[:, j] + h[kd[:, j]] * pd[:, j])
        nu[i - 1] = -np.log(P[i] / h[idx_m]) / dt
    
    r = xval + np.tile(nu, (nx+1, 1))
    output = {"r": r, "k": k, "ku": ku, "kd": kd, "pm": pm, "pu": pu, "pd": pd, "t": t, "dt": dt}
    return output

#%% Construction of the hybrid tree base on the trinomial tree for the interest rate

def hybrid_tree(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1):
    Tree_rate = r_tree(T, nt, kap, sg_r, b0, b10, b11, c1)
    
    dt = T / nt
    t  = np.linspace(0, T, (nt+1))
    nr = Tree_rate["r"].shape[0] - 1
    
    r_all = Tree_rate["r"]
    k = Tree_rate["k"]
    ku = Tree_rate["ku"]
    kd = Tree_rate["kd"]
    pm = Tree_rate["pm"]
    pu = Tree_rate["pu"]
    pd = Tree_rate["pd"]
    
    
    S = np.zeros((6**nt,nt+1))
    S[0,0] = S0
    r = np.zeros((6**nt,nt+1))
    k_r = np.zeros((6**nt,nt+1), dtype=int)
    k_r[0,0] = nr // 2
    r[0,0] = r_all[k_r[0,0], 0]
    k_mother = np.zeros((6**nt,nt), dtype=int)
    link = np.zeros((6**nt,nt+1), dtype=int)
    p = np.zeros((6**nt,nt))
    
    print("End of initialization")
    print("Beginning of the loop on time steps")
    
    for i in range(nt):
        num_ind = 6**i
        print("t =", i," with ", num_ind, " states")
        for j in range(num_ind):
            
            k_r[6*j,i+1] = ku[k_r[j,i],i]
            k_r[6*j+1,i+1] = k[k_r[j,i],i]
            k_r[6*j+2,i+1] = kd[k_r[j,i],i]
            k_r[6*j+3,i+1] = ku[k_r[j,i],i]
            k_r[6*j+4,i+1] = k[k_r[j,i],i]
            k_r[6*j+5,i+1] = kd[k_r[j,i],i]
            
            r[6*j,i+1] = r_all[k_r[6*j,i+1],i+1]
            r[6*j+1,i+1] = r_all[k_r[6*j+1,i+1],i+1]
            r[6*j+2,i+1] = r_all[k_r[6*j+2,i+1],i+1]
            r[6*j+3,i+1] = r_all[k_r[6*j+3,i+1],i+1]
            r[6*j+4,i+1] = r_all[k_r[6*j+4,i+1],i+1]
            r[6*j+5,i+1] = r_all[k_r[6*j+5,i+1],i+1]
            
            k_mother[6*j,i] = j
            k_mother[6*j+1,i] = j
            k_mother[6*j+2,i] = j
            k_mother[6*j+3,i] = j
            k_mother[6*j+4,i] = j
            k_mother[6*j+5,i] = j
            
            link[6*j,i+1] = 36*j
            link[6*j+1,i+1] = 36*j+6
            link[6*j+2,i+1] = 36*j+12
            link[6*j+3,i+1] = 36*j+18
            link[6*j+4,i+1] = 36*j+24
            link[6*j+5,i+1] = 36*j+30
            
            u = np.exp((r[j,i] - sg_S**2/2)*dt + sg_S*np.sqrt(dt))
            d = np.exp((r[j,i] - sg_S**2/2)*dt - sg_S*np.sqrt(dt))
            puS = (np.exp(r[j,i]*dt) - d) / (u - d)
            pdS = 1 - puS
            
            if rho < 0:
                p[6*j,i] = pu[k_r[j,i],i]*puS*(1+rho)
                p[6*j+1,i] = pm[k_r[j,i],i]*puS - rho/4 * (pu[k_r[j,i],i]*puS + pd[k_r[j,i],i]*pdS)
                p[6*j+2,i] = pd[k_r[j,i],i]*puS - rho/4 * (pu[k_r[j,i],i]*puS + pd[k_r[j,i],i]*pdS)
                p[6*j+3,i] = pu[k_r[j,i],i]*pdS - rho/4 * (pu[k_r[j,i],i]*puS + pd[k_r[j,i],i]*pdS)
                p[6*j+4,i] = pm[k_r[j,i],i]*pdS - rho/4 * (pu[k_r[j,i],i]*puS + pd[k_r[j,i],i]*pdS)
                p[6*j+5,i] = pd[k_r[j,i],i]*pdS*(1+rho)
            else:
                p[6*j,i] = pu[k_r[j,i],i]*puS + rho/4 * (pd[k_r[j,i],i]*puS + pu[k_r[j,i],i]*pdS)
                p[6*j+1,i] = pm[k_r[j,i],i]*puS + rho/4 * (pd[k_r[j,i],i]*puS + pu[k_r[j,i],i]*pdS)
                p[6*j+2,i] = pd[k_r[j,i],i]*puS*(1-rho)
                p[6*j+3,i] = pu[k_r[j,i],i]*pdS*(1-rho)
                p[6*j+4,i] = pm[k_r[j,i],i]*pdS + rho/4 * (pd[k_r[j,i],i]*puS + pu[k_r[j,i],i]*pdS)
                p[6*j+5,i] = pd[k_r[j,i],i]*pdS + rho/4 * (pd[k_r[j,i],i]*puS + pu[k_r[j,i],i]*pdS)
            
            S[6*j,i+1] = S[j,i]*u
            S[6*j+1,i+1] = S[j,i]*u
            S[6*j+2,i+1] = S[j,i]*u
            S[6*j+3,i+1] = S[j,i]*d
            S[6*j+4,i+1] = S[j,i]*d
            S[6*j+5,i+1] = S[j,i]*d
            
            if j%1000 == 0:
                print("End of state ",j)
                
    print("End of the construction of the hybrid tree")
    print("__________________________________________")
                
    output = {"S": S, "r": r, "k_mother": k_mother, "link": link, "p": p, "t": t, "dt": dt}
    return output

#%% Functions to compute the option fair value on the tree (European and Bermudan style)

def option_tree_EU(T, nt, kap, sg_r, sg_S, rho, S0, K, b0, b10, b11, c1, payoff_type="put"):
    Tree_hybrid = hybrid_tree(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1)
    
    S = Tree_hybrid["S"]
    r = Tree_hybrid["r"]
    l = Tree_hybrid["link"]
    p = Tree_hybrid["p"]
    
    dt = T / nt
    t  = np.linspace(0, T, (nt+1))
    nS = S.shape[0]
    h  = np.zeros((nS,nt+1))
    
    if payoff_type=="put":
        h[:,-1] = np.maximum(K - S[:,-1],0)
    elif payoff_type=="call":
        h[:,-1] = np.maximum(S[:,-1] - K,0)
    
    for i in range(nt-1, -1, -1):
        num_ind = 6**i
        idx = l[:num_ind,i]
        
        h[:num_ind,i] = np.exp(-r[:num_ind,i] * dt) * (h[idx,i+1]*p[idx,i] + 
                        h[idx+1,i+1]*p[idx+1,i] + h[idx+2,i+1]*p[idx+2,i] +
                        h[idx+3,i+1]*p[idx+3,i] + h[idx+4,i+1]*p[idx+4,i] +
                        h[idx+5,i+1]*p[idx+5,i])
    
    output = {"P0":h[0,0], "P":h, "S": S, "r": r, "l": l, "p": p, "t":t, "dt":dt}
    return output

def option_tree_Berm(T, nt, kap, sg_r, sg_S, rho, S0, K, b0, b10, b11, c1, payoff_type="put"):
    Tree_hybrid = hybrid_tree(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1)
    
    S = Tree_hybrid["S"]
    r = Tree_hybrid["r"]
    l = Tree_hybrid["link"]
    p = Tree_hybrid["p"]
    
    dt = T / nt
    t  = np.linspace(0, T, (nt+1))
    nS = S.shape[0]
    h  = np.zeros((nS,nt+1))
    
    if payoff_type=="put":
        h[:,-1] = np.maximum(K - S[:,-1],0)
    elif payoff_type=="call":
        h[:,-1] = np.maximum(S[:,-1] - K,0)
    
    for i in range(nt-1, -1, -1):
        num_ind = 6**i
        idx = l[:num_ind,i]
        
        if t[i] % 1 == 0 and t[i] != 0:
            if payoff_type=="put":
                payoff = np.maximum(K - S[:num_ind,i],0)
            elif payoff_type=="call":
                payoff = np.maximum(S[:num_ind,i] - K,0)
            h_new = np.exp(-r[:num_ind,i] * dt) * (h[idx,i+1]*p[idx,i] + 
                        h[idx+1,i+1]*p[idx+1,i] + h[idx+2,i+1]*p[idx+2,i] +
                        h[idx+3,i+1]*p[idx+3,i] + h[idx+4,i+1]*p[idx+4,i] +
                        h[idx+5,i+1]*p[idx+5,i])
            h[:num_ind,i] = np.maximum(payoff,h_new)
        else:  
            h[:num_ind,i] = np.exp(-r[:num_ind,i] * dt) * (h[idx,i+1]*p[idx,i] + 
                        h[idx+1,i+1]*p[idx+1,i] + h[idx+2,i+1]*p[idx+2,i] +
                        h[idx+3,i+1]*p[idx+3,i] + h[idx+4,i+1]*p[idx+4,i] +
                        h[idx+5,i+1]*p[idx+5,i])
    
    output = {"P0":h[0,0], "P":h, "S": S, "r": r, "l": l, "p": p, "t":t, "dt":dt}
    return output

#%% Main

#%% Definition of parameters

# Nelson-Siegel parameters
b0  =  0.03974889670677964
b10 =  0.0014721430763244212
b11 = -0.01693616284948873
c1  =  0.4222897254760573

# Parameters for Hull-White tree
T    =  2    # Time horizon in years
nt   =  4    # Number of time steps
kap  =  0.1  # Speed of mean reversion
sg_r =  0.02 # Volatility of short-term rates
sg_S =  0.04 # Volatility of the stock
rho  = -0.4  # Correlation between rates and stock
S0   =  90   # Initial value of the stock
K    =  100  # Strike price of the option

#%% Construction of the hybrid tree and computation of the option fair value

Tree_option = option_tree_Berm(T, nt, kap, sg_r, sg_S, rho, S0, K, b0, b10, b11, c1, payoff_type="put")

#%% Display of the results

t = Tree_option["t"]
dt = Tree_option["dt"]
P = Tree_option["P"]
P0 = Tree_option["P0"]
S = Tree_option["S"]
r = Tree_option["r"]
l = Tree_option["l"]
p = Tree_option["p"]

print("Price in t=0:",P0)

#%% Sparse version of the trees: do not solve issues...

def hybrid_tree_sparse(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1):
    Tree_rate = r_tree(T, nt, kap, sg_r, b0, b10, b11, c1)
    
    dt = T / nt
    t = np.linspace(0, T, nt+1)
    nr = Tree_rate["r"].shape[0] - 1
    
    r_all = Tree_rate["r"]
    k = Tree_rate["k"]
    ku = Tree_rate["ku"]
    kd = Tree_rate["kd"]
    pm = Tree_rate["pm"]
    pu = Tree_rate["pu"]
    pd = Tree_rate["pd"]
    
    size = 6**nt
    S = csr_matrix((size, nt+1), dtype=np.float64)
    S[0, 0] = S0
    r = csr_matrix((size, nt+1), dtype=np.float64)
    k_r = csr_matrix((size, nt+1), dtype=int)
    k_r[0, 0] = nr // 2
    r[0, 0] = r_all[k_r[0, 0], 0]
    k_mother = csr_matrix((size, nt), dtype=int)
    link = csr_matrix((size, nt+1), dtype=int)
    p = csr_matrix((size, nt), dtype=np.float64)
    
    print("End of initialization")
    print("Beginning of the loop on time steps")
    
    for i in range(nt):
        num_ind = 6**i
        print("t = ", i," with ", num_ind, " states")
        for j in range(num_ind):
            
            k_r[6*j, i+1] = ku[k_r[j, i], i]
            k_r[6*j+1, i+1] = k[k_r[j, i], i]
            k_r[6*j+2, i+1] = kd[k_r[j, i], i]
            k_r[6*j+3, i+1] = ku[k_r[j, i], i]
            k_r[6*j+4, i+1] = k[k_r[j, i], i]
            k_r[6*j+5, i+1] = kd[k_r[j, i], i]
            
            r[6*j, i+1] = r_all[k_r[6*j, i+1], i+1]
            r[6*j+1, i+1] = r_all[k_r[6*j+1, i+1], i+1]
            r[6*j+2, i+1] = r_all[k_r[6*j+2, i+1], i+1]
            r[6*j+3, i+1] = r_all[k_r[6*j+3, i+1], i+1]
            r[6*j+4, i+1] = r_all[k_r[6*j+4, i+1], i+1]
            r[6*j+5, i+1] = r_all[k_r[6*j+5, i+1], i+1]
            
            k_mother[6*j, i] = j
            k_mother[6*j+1, i] = j
            k_mother[6*j+2, i] = j
            k_mother[6*j+3, i] = j
            k_mother[6*j+4, i] = j
            k_mother[6*j+5, i] = j
            
            link[6*j, i+1] = 36 * j
            link[6*j+1, i+1] = 36 * j + 6
            link[6*j+2, i+1] = 36 * j + 12
            link[6*j+3, i+1] = 36 * j + 18
            link[6*j+4, i+1] = 36 * j + 24
            link[6*j+5, i+1] = 36 * j + 30
            
            u = np.exp((r[j, i] - sg_S**2 / 2) * dt + sg_S * np.sqrt(dt))
            d = np.exp((r[j, i] - sg_S**2 / 2) * dt - sg_S * np.sqrt(dt))
            puS = (np.exp(r[j, i] * dt) - d) / (u - d)
            pdS = 1 - puS
            
            if rho < 0:
                p[6*j, i] = pu[k_r[j, i], i] * puS * (1 + rho)
                p[6*j+1, i] = pm[k_r[j, i], i] * puS - rho / 4 * (pu[k_r[j, i], i] * puS + pd[k_r[j, i], i] * pdS)
                p[6*j+2, i] = pd[k_r[j, i], i] * puS - rho / 4 * (pu[k_r[j, i], i] * puS + pd[k_r[j, i], i] * pdS)
                p[6*j+3, i] = pu[k_r[j, i], i] * pdS - rho / 4 * (pu[k_r[j, i], i] * puS + pd[k_r[j, i], i] * pdS)
                p[6*j+4, i] = pm[k_r[j, i], i] * pdS - rho / 4 * (pu[k_r[j, i], i] * puS + pd[k_r[j, i], i] * pdS)
                p[6*j+5, i] = pd[k_r[j, i], i] * pdS * (1 + rho)
            else:
                p[6*j, i] = pu[k_r[j, i], i] * puS + rho / 4 * (pd[k_r[j, i], i] * puS + pu[k_r[j, i], i] * pdS)
                p[6*j+1, i] = pm[k_r[j, i], i] * puS + rho / 4 * (pd[k_r[j, i], i] * puS + pu[k_r[j, i], i] * pdS)
                p[6*j+2, i] = pd[k_r[j, i], i] * puS * (1 - rho)
                p[6*j+3, i] = pu[k_r[j, i], i] * pdS * (1 - rho)
                p[6*j+4, i] = pm[k_r[j, i], i] * pdS + rho / 4 * (pd[k_r[j, i], i] * puS + pu[k_r[j, i], i] * pdS)
                p[6*j+5, i] = pd[k_r[j, i], i] * pdS + rho / 4 * (pd[k_r[j, i], i] * puS + pu[k_r[j, i], i] * pdS)
            
            S[6*j, i+1] = S[j, i] * u
            S[6*j+1, i+1] = S[j, i] * u
            S[6*j+2, i+1] = S[j, i] * u
            S[6*j+3, i+1] = S[j, i] * d
            S[6*j+4, i+1] = S[j, i] * d
            S[6*j+5, i+1] = S[j, i] * d
            
            if j%1000 == 0:
                print("End of state ",j)
    
    output = {"S": S, "r": r, "k_mother": k_mother, "link": link, "p": p, "t": t, "dt": dt}
    return output

def option_tree_EU_sparse(T, nt, kap, sg_r, sg_S, rho, S0, K, b0, b10, b11, c1, payoff_type="put"):
    Tree_hybrid = hybrid_tree_sparse(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1)
    
    S = Tree_hybrid["S"]
    r = Tree_hybrid["r"]
    l = Tree_hybrid["link"]
    p = Tree_hybrid["p"]
    
    dt = T / nt
    t = np.linspace(0, T, nt+1)
    nS = S.shape[0]
    h = csr_matrix((nS, nt+1), dtype=np.float64)
    
    S_last_column = S[:, -1].toarray().flatten()
    
    if payoff_type == "put":
        h[:, -1] = np.maximum(K - S_last_column, 0)
    elif payoff_type == "call":
        h[:, -1] = np.maximum(S_last_column - K, 0)
    
    for i in range(nt-1, -1, -1):
        num_ind = 6**i
        idx = l[:num_ind, i].toarray()
        
        h[:num_ind, i] = np.exp(-r[:num_ind, i].toarray() * dt) * (
            h[idx, i+1].toarray() * p[idx, i].toarray() + 
            h[idx+1, i+1].toarray() * p[idx+1, i].toarray() + 
            h[idx+2, i+1].toarray() * p[idx+2, i].toarray() + 
            h[idx+3, i+1].toarray() * p[idx+3, i].toarray() + 
            h[idx+4, i+1].toarray() * p[idx+4, i].toarray() + 
            h[idx+5, i+1].toarray() * p[idx+5, i].toarray()
        )
    
    output = {"P0":h[0,0], "P":h, "S": S, "r": r, "l": l, "p": p, "t":t, "dt":dt}
    return output

def option_tree_Berm_sparse(T, nt, kap, sg_r, sg_S, rho, S0, K, b0, b10, b11, c1, payoff_type="put"):
    Tree_hybrid = hybrid_tree_sparse(T, nt, kap, sg_r, sg_S, rho, S0, b0, b10, b11, c1)
    
    S = Tree_hybrid["S"]
    r = Tree_hybrid["r"]
    l = Tree_hybrid["link"]
    p = Tree_hybrid["p"]
    
    dt = T / nt
    t  = np.linspace(0, T, (nt+1))
    nS = S.shape[0]
    h  = csr_matrix((nS, nt+1), dtype=np.float64)
    
    S_last_column = S[:, -1].toarray().flatten()
    
    if payoff_type=="put":
        h[:,-1] = np.maximum(K - S_last_column,0)
    elif payoff_type=="call":
        h[:,-1] = np.maximum(S_last_column - K,0)
    
    for i in range(nt-1, -1, -1):
        num_ind = 6**i
        idx = l[:num_ind,i].toarray()
        
        if t[i] % 1 == 0 and t[i] != 0:
            S_column_i = S[:num_ind,i].toarray().flatten()
            if payoff_type=="put":
                payoff = np.maximum(K - S_column_i,0).reshape(-1,1)
            elif payoff_type=="call":
                payoff = np.maximum(S_column_i - K,0).reshape(-1,1)
                
            h_new = np.exp(-r[:num_ind,i].toarray() * dt) * (
                h[idx,i+1].toarray()*p[idx,i].toarray() + 
                h[idx+1,i+1].toarray()*p[idx+1,i].toarray() + 
                h[idx+2,i+1].toarray()*p[idx+2,i].toarray() +
                h[idx+3,i+1].toarray()*p[idx+3,i].toarray() + 
                h[idx+4,i+1].toarray()*p[idx+4,i].toarray() +
                h[idx+5,i+1].toarray()*p[idx+5,i].toarray())
            h[:num_ind,i] = np.maximum(payoff,h_new)
        else:  
            h[:num_ind,i] =  np.exp(-r[:num_ind,i].toarray() * dt) * (
                h[idx,i+1].toarray()*p[idx,i].toarray() + 
                h[idx+1,i+1].toarray()*p[idx+1,i].toarray() + 
                h[idx+2,i+1].toarray()*p[idx+2,i].toarray() +
                h[idx+3,i+1].toarray()*p[idx+3,i].toarray() + 
                h[idx+4,i+1].toarray()*p[idx+4,i].toarray() +
                h[idx+5,i+1].toarray()*p[idx+5,i].toarray())
    
    output = {"P0":h[0,0], "P":h, "S": S, "r": r, "l": l, "p": p, "t":t, "dt":dt}
    return output
