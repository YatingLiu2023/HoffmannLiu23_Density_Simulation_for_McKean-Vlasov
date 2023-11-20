#!/usr/bin/env python
# coding: utf-8

# # Density simulation of a McKean-Vlasov SDE with linear interaction
# 
# This notebook implements the simulations described in Section 3.1 of 
# 
# - Hoffmann, M. and Liu, Y (2023). A statistical approach for simulating the density solution of a McKean-Vlasov equation. 
# 
# -------------------------------
# 
# **Contents**
# 
# * [1. Definition of the McKean-Vlasov equation, the particle system and the Euler scheme](#chapter1)
# 
# * [2. Density simulation with Gaussian-based high order Kernels](#chapter2)
# 
#     * [2.1 Definition of Gaussian-based high order Kernels](#section2-1)
#     
#     * [2.2 Gaussian Kernel ( Kernel Order = 1 )](#section2-2)
#     
#     * [2.3 Kernel Order = 3](#section2-3)
#     
#     * [2.4 Kernel Order = 5](#section2-4)
#     
#     * [2.5 Kernel Order = 7](#section2-5)
#     
#     * [2.6 Kernel Order = 9](#section2-6)
#     
#     * [2.7 Comparison of simulation results with different kernels](#section2-7)
# 
# -------------------------------

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import pandas as pd 
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


# ## 1. Definition of the McKean-Vlasov equation, the particle system and the Euler scheme <a class="anchor" id="chapter1"></a> 
# 

# We consider a stochastic process $(X_t)_{t\in[0, T]}$ defined by the following McKean-Vlasov SDE with linear interaction 
# 
# $$dX_t = - \int_{\mathbb{R}}(X_t-x)\mu_t(dx)dt+dB_t\quad \text{with}\quad X_0\sim \mathcal{N}(3, \frac{1}{2}),\hspace{2cm}(1)$$
# 
# where for every  $t\in[0,T]$, $\mu_t$ denotes the probability distrubtion of $X_t$.  This process $(X_t)_{t\in[0, T]}$ is a stationary Ornstein-Uhlenbeck process, namely, for every $t\in[0,T], \; \mu_t=\mathcal{N}(3, \frac{1}{2})$.

# Let $N$ be the number of particles. The $N$-particle system $(X_t^{1}, ..., X_t^{N})_{t\in[0,T]}$ corresponding to Equation (1) is defined as follows :
# 
# 
# \begin{equation}
# dX_t^{n}=- \frac{1}{N} \sum_{i=1}^{N}\big(X_t^{n}-X_t^{i}\big) dt+\sigma B_t^{n}.\hspace{2cm}(2)
# \end{equation}

# Let $M$ denote the time discretization number for the Euler scheme. Set $h=\frac{T}{M}$ and $t_m=m\cdot h, 0\leq m\leq M$. Combining the particle system (2) and the Euler scheme, we obtain the following discrete particle system :
# 
# \begin{equation}
# X^{n}_{t_{m+1}}=X^{n}_{t_{m}}+h\cdot  \frac{1}{N}\sum_{i=1}^{N}\Big(X_{t_m}^{i}-X_{t_m}^{n}\Big)+\sqrt{h} Z_{m+1}^{n},\quad 1\leq n\leq N,
# \end{equation}
# where $ \: Z_{m+1}^{n}:=\frac{1}{\sqrt{h}}(B^n_{t_{m+1}}-B^n_{t_{m}})$ are i.i.d random variables having the standard normal distribution $\mathcal{N}(0,1)$.

# * In this notebook, we fix $T=1$ and $M=100$.

# In[2]:


# Parameters

T=1. 
M=100  # Time discretization number
h=T/M  # Time step

m_X0=3  # Mean of the distribution of X_0


# In the next cell, the function `Euler_one_step` defines the operator of the Euler scheme for one time step.

# In[3]:


def Euler_one_step(X_in,N_in):
    nn=np.ones(N_in)/N_in
    EXin=np.dot(nn,X_in)
    X_out=X_in+h*(EXin-X_in)+np.sqrt(h)*np.random.normal(0, 1, N_in)
    return X_out


# We consider different particle numbers : $N\in\{2^7=128, 2^8=256, ..., 2^{15}=32768\}$. 

# In[4]:


pn=9
N_power=np.linspace(7,15,pn) 
print (N_power)

N_vec=2**N_power.astype(int) # Number of particles, from 2^7 to 2^15
print (N_vec)


# First, we compute and save the particle systems at time T=1 with different particle numbers in `N_vec`. For each given particle number, we implement 30 identical and independent simulations.

# In[9]:


# Number of identical and independent simulations 

NNs=30


# In[5]:


for i in tqdm(range(pn)):
    for nns in range(NNs):
        particle_process=np.zeros((N_vec[i],M))
        particle_process[:,0]=np.random.normal(m_X0, np.sqrt(1/2), N_vec[i])

        for m in range(M-1):
            particle_process[:,m+1]=Euler_one_step(particle_process[:,m],N_vec[i])
            
        np.savetxt("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", particle_process[:,-1], delimiter=",")


# The following cell defines the true density function of $X_t$, for every $t\in[0,T]$, which is the density function of the normal distribution $\mathcal{N}(3,\frac{1}{2})$.

# In[10]:


# Definition of the true density function 

def density_true(x_in):
    return norm.pdf(x_in,m_X0,np.sqrt(1/2))


# -------------------
# 
# 
# # 2. Density simulation with Gaussian-based high order Kernel <a class="anchor" id="chapter2"></a> 

# In this section, we simulate the density of $X_t$ with the optimal bandwidth choice defined in Corollary 2.11 of [Hoffmann, M. and Liu, Y (2023)]
# 
# $$\eta_{\text{opt}}=N^{-\frac{1}{2(l+1)+1}},$$
# 
# where $N$ is the number of particles and $l$ is the order of the Kernel. The error $\mathbb{E}\Big[\,\big|\,\widehat{\mu}_T^{N, h, \eta}(x)-\mu_T(x)\,\big|^2\Big]$ is approximated by 
# 
# $$\varepsilon_N=\frac{1}{30}\sum_{j=1}^{30}\max_{x\in\mathcal{D}}\Big|\;\big(\widehat{\mu}_T^{N, h, \eta}\big)_j(x)-\mu_T(x)\;\Big|^{\,2},$$
# 
# where $\big(\widehat{\mu}_T^{N, h, \eta}\big)_j, \, 1\leq j\leq 30$ are simulated density functions based on the above 30 independent particle systems and the domain $\mathcal{D}$ is a uniform grid of 1000 points in [0,6]. 

# ## 2.1 Definition of Gaussian-based high order kernel <a class="anchor" id="section2-1"></a> 
# 
# 
# 
# The Gaussian-based high order kernels that we use in this notebook are 
# - Order 1 : $\phi(x)=\frac{1}{\sqrt{2\pi}}\exp\big(-\frac{x^2}{2}\big)$
# 
# 
# 
# - Order 3 : $\frac{1}{2}(3-x^2)\,\phi(x)$
# 
# 
# 
# - Order 5 : $\frac{1}{8}(15-10x^2+x^4)\,\phi(x)$
# 
# 
# 
# - Order 7 : $\frac{1}{48}(105-105x^2+21x^4-x^6)\,\phi(x)$
# 
# 
# 
# - Order 9 : $\frac{1}{384}(945-1260x^2+378x^4-36x^6+x^8)\,\phi(x)$
# 
# 
# which are defined in the following paper (see Table 1):
# 
# - Wand, Matthew P., and William R. Schucany. "Gaussian‐based kernels." Canadian Journal of Statistics 18, no. 3 (1990): 197-204.
# 
# 

# In[11]:


# Kernel order 1

def kernel_O1(x_in):
    return np.exp(-0.5*(x_in**2))/np.sqrt(2*np.pi)

kernel_O1_fun = lambda x: np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)

# Kernel order 3

def kernel_O3(x_in):
    return 0.5*(3-x_in**2)*np.exp(-0.5*(x_in**2))/np.sqrt(2*np.pi)

kernel_O3_fun = lambda x: 0.5*(3-x**2)*np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)

# Kernel order 5

def kernel_O5(x_in):
    return (15-10*x_in**2+x_in**4)*np.exp(-0.5*(x_in**2))/np.sqrt(2*np.pi)/8

kernel_O5_fun = lambda x: (15-10*x**2+x**4)*np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)/8

# Kernel order 7

def kernel_O7(x_in):
    return (105-105*x_in**2+21*x_in**4-x_in**6)*np.exp(-0.5*(x_in**2))/np.sqrt(2*np.pi)/48

kernel_O7_fun = lambda x: (105-105*x**2+21*x**4-x**6)*np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)/48

# Kernel order 9

def kernel_O9(x_in):
    return (945-1260*x_in**2+378*x_in**4-36*x_in**6+x_in**8)*np.exp(-0.5*(x_in**2))/np.sqrt(2*np.pi)/384

kernel_O9_fun = lambda x: (945-1260*x**2+378*x**4-36*x**6+x**8)*np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)/384


# In[12]:


test_xx=np.linspace(-5,5,1000)
Kernel_O1_plot=kernel_O1_fun(test_xx)
Kernel_O3_plot=kernel_O3_fun(test_xx)
Kernel_O5_plot=kernel_O5_fun(test_xx)
Kernel_O7_plot=kernel_O7_fun(test_xx)
Kernel_O9_plot=kernel_O9_fun(test_xx)


fig, axs = plt.subplots(3, 2, figsize=(10,8))
axs[0, 0].plot(test_xx, Kernel_O1_plot)
axs[0, 0].set_title('Kernel order 1')
axs[0, 1].plot(test_xx, Kernel_O3_plot)
axs[0, 1].set_title('Kernel order 3')
axs[1, 0].plot(test_xx, Kernel_O5_plot, 'tab:orange')
axs[1, 0].set_title('Kernel order 5')
axs[1, 1].plot(test_xx, Kernel_O7_plot, 'tab:green')
axs[1, 1].set_title('Kernel order 7')
axs[2, 0].plot(test_xx, Kernel_O9_plot, 'tab:red')
axs[2, 0].set_title('Kernel order 9')
axs[2, 1].axis('off')
fig.tight_layout()


# ## 2.2 Gaussian Kernel ( Kernel Order = 1 ) <a class="anchor" id="section2-2"></a> 
# 

# In[13]:


error_N_Gaussian_kernel=[] 

l_regularity_plus_one=2 # l=1 then l+1=2

for i in tqdm(range(pn)):
    error=np.zeros(NNs)
    for nns in range(NNs):

        density_particle= np.array(pd.read_csv("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", sep=',',header=None))
        
        Error_simu_number=1000
        
        X_simu = np.linspace(0, 6, Error_simu_number)[:, np.newaxis]
        
        Error_simu_vec=np.zeros(Error_simu_number)
        error_vec=np.zeros(Error_simu_number)
        
        eta_opt = (N_vec[i])**(-1./(2*l_regularity_plus_one+1))
        
        for j in range(Error_simu_number):
            Error_simu_vec[j]=kernel_O1_fun((X_simu[j]-density_particle)/eta_opt).sum()/(N_vec[i]*eta_opt) # O2!  
            error_vec[j]=np.abs(Error_simu_vec[j] - density_true(X_simu[j]))

        error_one=error_vec.max()**2
        error[nns]=error_one
        
    print ("i=",i,";", "the number of particle is", N_vec[i],";", "The optimal bandwidth is",eta_opt, ".")
    print (i,"the error is :", error.mean())
    
    error_N_Gaussian_kernel.append(error.mean()) # Name of the error vector !

    np.savetxt("error_"+str(i)+"_op_Gaussian_kernel.csv", error_N_Gaussian_kernel, delimiter=",") # Name of the error vector !



# ---------------------------------------
# 
# The next cell shows the log-log error curve of the density simulation with the Gaussian kernel. 

# In[14]:


plt.figure(figsize=(8,6))
plt.plot(np.log2(N_vec),np.log2(error_N_Gaussian_kernel),"r")


# The least-square estimate of the slope of the above curve is the following. 

# In[15]:


X=np.log2(N_vec).reshape(-1,1)
y=np.log2(error_N_Gaussian_kernel)
reg_Gaussian = LinearRegression().fit(X, y)
print ("The estimate of the slope is", reg_Gaussian.coef_)
print ("The estimate of the intercept is", reg_Gaussian.intercept_)


# ## 2.3 Kernel order = 3 <a class="anchor" id="section2-3"></a> 

# In[16]:


error_N_O3_kernel=[] 

l_regularity_plus_one=4  # l=1 then l+1=2

for i in tqdm(range(pn)):
    error=np.zeros(NNs)
    for nns in range(NNs):

        density_particle= np.array(pd.read_csv("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", sep=',',header=None))
        
        Error_simu_number=1000
        
        X_simu = np.linspace(0, 6, Error_simu_number)[:, np.newaxis]
        
        Error_simu_vec=np.zeros(Error_simu_number)
        error_vec=np.zeros(Error_simu_number)
        
        eta_opt = (N_vec[i])**(-1./(2*l_regularity_plus_one+1))
        
        for j in range(Error_simu_number):
            Error_simu_vec[j]=kernel_O3_fun((X_simu[j]-density_particle)/eta_opt).sum()/(N_vec[i]*eta_opt) 
            error_vec[j]=np.abs(Error_simu_vec[j] - density_true(X_simu[j]))

        error_one=error_vec.max()**2
        error[nns]=error_one
        
    print ("i=",i,";", "the number of particle is", N_vec[i],";", "The optimal bandwidth is",eta_opt, ".")
    print (i,"the error is :", error.mean())
    
    error_N_O3_kernel.append(error.mean()) 

    np.savetxt("error_"+str(i)+"_op_Order_3_kernel.csv", error_N_O3_kernel, delimiter=",") 



# ---------------------------------------
# 
# The next cell shows the log-log error curve of the density simulation with the Gaussian high order kernel of order 3. 

# In[17]:


plt.figure(figsize=(8,6))

plt.plot(np.log2(N_vec),np.log2(error_N_O3_kernel),"r")


# The least-square estimate of the slope of the above curve is the following. 

# In[18]:


X=np.log2(N_vec).reshape(-1,1)
y=np.log2(error_N_O3_kernel)
reg_Kernal_O3 = LinearRegression().fit(X, y)
print ("The estimate of the slope is", reg_Kernal_O3.coef_)
print ("The estimate of the intercept is", reg_Kernal_O3.intercept_)


# ## 2.4 Kernel order = 5 <a class="anchor" id="section2-4"></a> 

# In[19]:


error_N_O5_kernel=[] 

l_regularity_plus_one=6 

for i in tqdm(range(pn)):
    error=np.zeros(NNs)
    for nns in range(NNs):

        density_particle= np.array(pd.read_csv("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", sep=',',header=None))
        
        Error_simu_number=1000
        
        X_simu = np.linspace(0, 6, Error_simu_number)[:, np.newaxis]
        
        Error_simu_vec=np.zeros(Error_simu_number)
        error_vec=np.zeros(Error_simu_number)
        
        eta_opt = (N_vec[i])**(-1./(2*l_regularity_plus_one+1))
        
        for j in range(Error_simu_number):
            Error_simu_vec[j]=kernel_O5_fun((X_simu[j]-density_particle)/eta_opt).sum()/(N_vec[i]*eta_opt) 
            error_vec[j]=np.abs(Error_simu_vec[j] - density_true(X_simu[j]))

        error_one=error_vec.max()**2
        error[nns]=error_one
        
    print ("i=",i,";", "the number of particle is", N_vec[i],";", "The optimal bandwidth is",eta_opt, ".")
    print (i,"the error is :", error.mean())
    
    error_N_O5_kernel.append(error.mean()) 

    np.savetxt("error_"+str(i)+"_op_Order_5_kernel.csv", error_N_O5_kernel, delimiter=",") 



# ---------------------------------------
# 
# The next cell shows the log-log error curve of the density simulation with the Gaussian high order kernel of order 5. 

# In[20]:


plt.figure(figsize=(8,6))

plt.plot(np.log2(N_vec),np.log2(error_N_O5_kernel),"r")


# The least-square estimate of the slope of the above curve is the following.

# In[21]:


X=np.log2(N_vec).reshape(-1,1)
y=np.log2(error_N_O5_kernel)
reg_Kernal_O5 = LinearRegression().fit(X, y)

print ("The estimate of the slope is", reg_Kernal_O5.coef_)
print ("The estimate of the intercept is", reg_Kernal_O5.intercept_)


# ## 2.5 Kernel order = 7 <a class="anchor" id="section2-5"></a> 

# In[22]:


error_N_O7_kernel=[] 

l_regularity_plus_one=8 

for i in tqdm(range(pn)):
    error=np.zeros(NNs)
    for nns in range(NNs):

        density_particle= np.array(pd.read_csv("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", sep=',',header=None))
        
        Error_simu_number=1000
        
        X_simu = np.linspace(0, 6, Error_simu_number)[:, np.newaxis]
        
        Error_simu_vec=np.zeros(Error_simu_number)
        error_vec=np.zeros(Error_simu_number)
        
        eta_opt = (N_vec[i])**(-1./(2*l_regularity_plus_one+1))
        
        for j in range(Error_simu_number):
            Error_simu_vec[j]=kernel_O7_fun((X_simu[j]-density_particle)/eta_opt).sum()/(N_vec[i]*eta_opt)   
            error_vec[j]=np.abs(Error_simu_vec[j] - density_true(X_simu[j]))

        error_one=error_vec.max()**2
        error[nns]=error_one
        
    print ("i=",i,";", "the number of particle is", N_vec[i],";", "The optimal bandwidth is",eta_opt, ".")
    print (i,"the error is :", error.mean())
    
    error_N_O7_kernel.append(error.mean()) 

    np.savetxt("error_"+str(i)+"_op_Order_7_kernel.csv", error_N_O7_kernel, delimiter=",") 



# ---------------------------------------
# 
# The next cell shows the log-log error curve of the density simulation with the Gaussian high order kernel of order 7. 

# In[23]:


plt.figure(figsize=(8,6))

plt.plot(np.log2(N_vec),np.log2(error_N_O7_kernel),"r")


# The least-square estimate of the slope of the above curve is the following.

# In[24]:


X=np.log2(N_vec).reshape(-1,1)
y=np.log2(error_N_O7_kernel)
reg_Kernal_O7 = LinearRegression().fit(X, y)


print ("The estimate of the slope is", reg_Kernal_O7.coef_)
print ("The estimate of the intercept is", reg_Kernal_O7.intercept_)


# ## 2.6 Kernel order = 9 <a class="anchor" id="section2-6"></a> 

# In[25]:


error_N_O9_kernel=[] 

l_regularity_plus_one=10 

for i in tqdm(range(pn)):
    error=np.zeros(NNs)
    for nns in range(NNs):

        density_particle= np.array(pd.read_csv("particle_num_"+str(N_vec[i])+"repeat"+str(nns)+".csv", sep=',',header=None))
        
        Error_simu_number=1000
        
        X_simu = np.linspace(0, 6, Error_simu_number)[:, np.newaxis]
        
        Error_simu_vec=np.zeros(Error_simu_number)
        error_vec=np.zeros(Error_simu_number)
        
        eta_opt = (N_vec[i])**(-1./(2*l_regularity_plus_one+1))
        
        for j in range(Error_simu_number):
            Error_simu_vec[j]=kernel_O9_fun((X_simu[j]-density_particle)/eta_opt).sum()/(N_vec[i]*eta_opt) 
            error_vec[j]=np.abs(Error_simu_vec[j] - density_true(X_simu[j]))

        error_one=error_vec.max()**2
        error[nns]=error_one
        
    print ("i=",i,";", "the number of particle is", N_vec[i],";", "The optimal bandwidth is",eta_opt, ".")
    print (i,"the error is :", error.mean())
    
    error_N_O9_kernel.append(error.mean())

    np.savetxt("error_"+str(i)+"_op_Order_9_kernel.csv", error_N_O9_kernel, delimiter=",") 



# ---------------------------------------
# 
# 
# The next cell shows the log-log error curve of the density simulation with the Gaussian high order kernel of order 9. 

# In[26]:


plt.figure(figsize=(8,6))

plt.plot(np.log2(N_vec),np.log2(error_N_O9_kernel),"r")


# The least-square estimate of the slope of the above curve is the following.

# In[27]:


X=np.log2(N_vec).reshape(-1,1)
y=np.log2(error_N_O9_kernel)
reg_Kernal_O9 = LinearRegression().fit(X, y)


print ("The estimate of the slope is", reg_Kernal_O9.coef_)
print ("The estimate of the intercept is", reg_Kernal_O9.intercept_)


# ## 2.7 Comparison of simulation results with different kernels <a class="anchor" id="section2-7"></a> 

# The next cell shows the log-log error curves of the density simulation with different Gaussian high order kernels for $l=1$ (purple), $l=3$ (blue), $l=5$ (red), $l=7$ (orange), $l=9$ (green). 

# In[38]:


plt.figure(figsize = (8,6))
plt.plot(np.log2(N_vec), np.log2(error_N_Gaussian_kernel),"purple") # Order 1
plt.plot(np.log2(N_vec), np.log2(error_N_O3_kernel),"b") # Order 3
plt.plot(np.log2(N_vec), np.log2(error_N_O5_kernel),"r") # Order 5
plt.plot(np.log2(N_vec), np.log2(error_N_O7_kernel),"orange") # Order 7
plt.plot(np.log2(N_vec), np.log2(error_N_O9_kernel),"green") # Order 9
plt.savefig('strong_error1.png')


# The next cell shows the slopes $a_l$ of the above figure as a function of the kernal order $l$. 

# In[71]:


slope_vec=np.array([reg_Gaussian.coef_,reg_Kernal_O3.coef_,reg_Kernal_O5.coef_,reg_Kernal_O7.coef_,reg_Kernal_O9.coef_])

plt.figure(figsize = (8,6))
plt.plot(np.array([1,3,5,7,9]),slope_vec,"o-")
plt.xticks([1,3,5,7,9])
plt.xlabel("Order of Gaussian Kernel", fontsize = 15)
plt.ylabel("Slope of the log-log error", fontsize = 15)

plt.xticks(fontsize = 15) 
plt.yticks(fontsize = 15) 


plt.savefig('strong_error_slope.eps', format='eps')



# In[ ]:





# In[72]:


plt.figure(figsize = (8,6))
plt.plot(np.log2(N_vec), np.log2(error_N_Gaussian_kernel),"purple", label="Order 1") # Order 1
plt.plot(np.log2(N_vec), np.log2(error_N_O3_kernel),"b", label="Order 3") # Order 3
plt.plot(np.log2(N_vec), np.log2(error_N_O5_kernel),"r", label="Order 5") # Order 5
plt.plot(np.log2(N_vec), np.log2(error_N_O7_kernel),"orange", label="Order 7") # Order 7
plt.plot(np.log2(N_vec), np.log2(error_N_O9_kernel),"green", label="Order 9") # Order 9
plt.legend(fontsize=12)


plt.xlabel("$\log_{2}N$", fontsize = 15)
plt.ylabel("$\log_{2}\mathcal{E}_N$", fontsize = 15)

plt.xticks(fontsize = 15) 
plt.yticks(fontsize = 15) 

plt.savefig('strong_error2.png')

plt.savefig('strong_error2.eps', format='eps')


# In[73]:


plt.figure(figsize = (8,6))
plt.plot(np.log2(N_vec), np.log2(error_N_Gaussian_kernel),"purple", dashes=[1, 5, 20, 5], label="Order 1") # Order 1
plt.plot(np.log2(N_vec), np.log2(error_N_O3_kernel),"b", linestyle = 'dashed', label="Order 3") # Order 3
plt.plot(np.log2(N_vec), np.log2(error_N_O5_kernel),"r",linestyle = 'dashdot',  label="Order 5") # Order 5
plt.plot(np.log2(N_vec), np.log2(error_N_O7_kernel),"orange",linestyle = 'dotted', label="Order 7") # Order 7
plt.plot(np.log2(N_vec), np.log2(error_N_O9_kernel),"green", linestyle = 'solid', label="Order 9") # Order 9
plt.legend(fontsize=12)


plt.xlabel("$\log_{2}N$", fontsize = 15)
plt.ylabel("$\log_{2}\mathcal{E}_N$", fontsize = 15)

plt.xticks(fontsize = 15) 
plt.yticks(fontsize = 15) 

plt.savefig('strong_error3.png')

plt.savefig('strong_error3.eps', format='eps')


# In[ ]:




