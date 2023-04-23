#!/usr/bin/env python
# coding: utf-8

# ### REDES NEURONALES 2021- Trabajo Práctico 1
# 

# Considere el modelo Integrate-and-Fire para la evolución temporal del potencial de membrana Vm(t) entre el interior y el exterior de una neurona gen ́erica, tal cual lo vimos en las clases tericas.
# 
# Primera parte: sin activación del mecanismo de disparo.
# Considere solo la ecuación diferencial del modelo, sin activar el mecanismo de disparo:
# 
# dVm(t)/dt =1/τm (EL − Vm(t) + Rm Ie(t)), (1)
# 
# donde
# • τm = 10 ms es el tiempo caracter ́ıstico de la membrana,
# • EL = −65 mV es el potencial en reposo,
# • Rm = 10MΩ es la resistencia
# • Ie(t) es una corriente eléctrica externa.
# A) Considere el caso en que Ie = 0. Haga un estudio geom ́etrico de la din ́amica de la ecuacion (1) indicando la dinámica para tiempos largos (t → ∞).

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# ### Apartado A

# In[46]:


tau = 10
E = -65
R = 10
I = 0
Vfix = E
V = np.linspace(-100,20,100)
def dotV(V):
    return (E-V+R*I)/tau
plt.plot(V,np.zeros(100),color='blue',linestyle='--')
plt.plot(V,(dotV)(V),color='pink')
plt.scatter([Vfix],[0],color='green')
plt.xlabel('V')
plt.ylabel('dv/dt')
plt.arrow(Vfix-20.0,0.0,10.0,0.0,head_width=0.5,head_length=2,fc='y',ec='y')
plt.arrow(Vfix+20.0,0.0,-10.0,0.0,head_width=0.5,head_length=2,fc='y',ec='y')
plt.title('Gráfico donde I_e=0')
plt.show()






# ### Apartado B

# In[47]:


tau = 10 #ms
E = -65 #mV
R = 10 #M.Ohms
I = 2 #nA
print ('I.R=','I*R')
Vfix = (E+I*R)
V = np.linspace(-100,20,100)
def dotV(V):
    return (E-V+R*I)/tau
plt.plot(V,np.zeros(100),color='blue',linestyle='--')
plt.plot(V,(dotV)(V),color='pink')
plt.scatter([Vfix],[0],color='r')
plt.xlabel('V')
plt.ylabel('dV/dt')
plt.arrow(Vfix-20.0,0.0,10.0,0.0,head_width=0.5,head_length=2,fc='green',ec='green')
plt.arrow(Vfix+20.0,0.0,-10.0,0.0,head_width=0.5,head_length=2,fc='green',ec='green')
plt.title('Gráfico donde I_e= 2nA')
plt.show()


# ### Apartado C

# ### Apartado D

# In[33]:


tau = 10 #ms
E = -65 #mV
R = 10 #M.Ohms
I = 2 # nA
Vfix = E+I*R
def Vexacta(t):
    return E + R*I*(1-np.exp(-t/tau))
tfunc = np.vectorize(Vexacta)
t = np.linspace(0,200,100) # ms
plt.plot(t,Vfix*np.ones(100),color='gray',linestyle='--',label='V*=E+IR')
plt.plot(t,E*np.ones(100),color='gray',linestyle=':',label='E')
plt.plot(t,np.vectorize(Vexacta)(t),color='r')
plt.xlabel('t[ms]')
plt.ylabel('V[mV]')
plt.title('Gráfico de la solución exacta')
plt.legend()
plt.show()


# ### Apartado E

# In[2]:


def rk4(f,x,t,h,p):
    """
    Calcula un paso de integración del método de Runge Kutta orden 4.
    
    Argumentos de entrada:
    
        f : R^n -> R^n
        x = x(t) : R^n
        t = tiempo : R
        h = paso de tiempo : R
        p = parametros : R^q        
        
    Retorna aproximacion numérica de
    
        x(t+h) : R^n

    # Ejemplos:
    """    
    k1 = f(x,t,p)
    k2 = f(x+0.5*h*k1,t+0.5*h,p)
    k3 = f(x+0.5*h*k2,t+0.5*h,p)
    k4 = f(x+h*k3,t+h,p)
    return x+h*(k1+2.0*k2+2.0*k3+k4)/6.0


# In[3]:


def integrador_ode(m,f,x0,a,b,k,p):
    """
    Integra numéricamente la ODE
    
        dx/dt = f(x,t)
        
    sobre el intervalo t:[a,b] usando k pasos de integración y el método m, bajo condicion inicial x(a)=x0.
    No es necesario que a<b.
    
    Argumentos de entrada:
    
        m = metodo de integracion (ej. euler, rk2, etc.)
        f : R^n -> R^n
        x0 = condicion inicial : R
        a = tiempo inicial : R
        b = tiempo final : R
        k = num. pasos de integracion : N
        p = parametros : R^q   
        
        Retorna:
    
        t : R^{k+1} , t_j = a+j*dt para j=0,1,...,k
        x : R^{k+1,n} , x_ij = x_i(t_j) para i=0,1,...,n-1 y j=0,1,...,k
        
    donde a+k*dt = b.
    """  
    assert k>0
    n = len(x0)
    h = (b-a)/k
    x = np.zeros((n,k+1))
    t = np.zeros(k+1)
    x[:,0] = x0
    t[0] = a
    for j in range(k):
        t[j+1] = t[j] + h
        x[:,j+1] = m(f,x[:,j],t[j],h,p)
    return t,x


# In[4]:


def f(x,t,p):   
    """
    x[0]=V
    p[0]= tau
    p[1]= E
    p[2]= R
    p[3]= I
    """
    return (p[1]-x[0]+p[2]*p[3])/p[0]


# In[50]:


tau = 10 #ms
E = -65 #mV
R = 10 #M.Ohms
I = 2 # nA
p = [tau,E,R,I]
tini = 0 # ms
tend = 200 #ms
x0 = [E]
h = 0.05 #ms
k = int((tend-tini)/h)
t,V = integrador_ode(rk4,f,x0,tini,tend,k,p)


# In[51]:


def Vexacta(t):
    return E+R*I*(1-np.exp(-t/tau))
tfunc = np.vectorize(Vexacta)
Vfix = E+I*R
plt.plot(t,Vfix*np.ones(len(t)),color='green',linestyle='--',label='V*=E+IR')
plt.plot(t,E*np.ones(len(t)),color='green',linestyle=':',label='E')
plt.plot(t,np.vectorize(Vexacta)(t),color='blue',label='V exacta')
plt.scatter(t,V[0,:],color='r',label='V RK4')
plt.xlabel('t[ms]')
plt.ylabel('V[mV]')
plt.title('Gráfico de la solución exacta y la aproximación numérica')
plt.legend()
plt.show()


# ### Apartado F

# In[5]:


def integrador_ode_condicionado(m,f,x0,a,b,k,p,g):
    """
    Integra numéricamente la ODE
    
        dx/dt = f(x,t)
        
    sobre el intervalo t:[a,b] usando k pasos de integración y el método m, bajo condicion inicial x(a)=x0.
    No es necesario que a<b.
    
    Argumentos de entrada:
    
        m = metodo de integracion (ej. euler, rk2, etc.)
        f : R^n -> R^n
        x0 = condicion inicial : R
        a = tiempo inicial : R
        b = tiempo final : R
        k = num. pasos de integracion : N
        p = parametros : R^q 
        g : R^n --> R^n = función condicionante.
    
    Retorna:
    
        t : R^{k+1} , t_j = a+j*dt para j=0,1,...,k
        x : R^{k+1,n} , x_ij = x_i(t_j) para i=0,1,...,n-1 y j=0,1,...,k
        
    donde a+k*dt = b.
    """  
    assert k>0
    n = len(x0)
    h = (b-a)/k
    x = np.zeros((n,k+1))
    t = np.zeros(k+1)
    x[:,0] = x0
    t[0] = a
    for j in range(k):
        t[j+1] = t[j] + h
        x[:,j+1] = m(f,x[:,j],t[j],h,p)
        g(j+1,x,t,p)
    return t,x


# In[19]:


def g(j,x,t,p):
    """
    x[:,:] donde x[0,j]=V[t_j] con 0=t_0, t_1=t_0+h,...
    p[0] = tau
    p[1] = E
    p[2] = R
    p[3] = I
    p[4] = V_u : potencial umbral.
    """
    if x[0,j]>p[4]: 
        x[0,j] = p[1] 


# In[54]:


tau = 10 #ms
E = -65 #mV
R = 10 #M.Ohms
I = 2 # nA
V_u = -50 # mV
p = [tau,E,R,I,V_u]
tini = 0 # ms
tend = 200 #ms
x0 = [E]
h = 0.05 #ms
k = int((tend-tini)/h)
t,V = integrador_ode_condicionado(rk4,f,x0,tini,tend,k,p,g)


# In[55]:


def Vexacta(t):
    return E+R*I*(1-np.exp(-t/tau))
tfunc = np.vectorize(Vexacta)
Vfix = E+I*R
plt.plot(t,Vfix*np.ones(len(t)),color='brown',linestyle='--',label='V*=E+IR')
plt.plot(t,E*np.ones(len(t)),color='brown',linestyle=':',label='E')
plt.plot(t,V_u*np.ones(len(t)),color='yellow',linestyle=':',label='V_u')
plt.plot(t,np.vectorize(Vexacta)(t),color='green',label='solucion sin disparo',linestyle='-.')
plt.plot(t,V[0,:],color='red',linestyle=':',label='V RK4 + disparo')
plt.xlabel('t[ms]')
plt.ylabel('V[mV]')
plt.title('Gráfico con incorporación del mecanismo de disparo')
plt.legend()
plt.show()


# ### Apartado G

# In[20]:


def fg(x,t,p):   
    """
    x[0]= V
    p[0]= tau
    p[1]= E
    p[2]= R
    p[3]= I: R --> R
    """
    return (p[1]-x[0]+p[2]*p[3](t))/p[0]


# In[21]:


I_0= 2.5 #nA
def I(t):
    return I_0*np.cos(t/30)
tau = 10 #ms
E = -65 #mV
R = 10 #M.Ohms
V_u = -50 # mV
p = [tau,E,R,I,V_u]
tini = 0 # ms
tend = 200 #ms
x0 = [E]
h = 0.05 #ms
k = int((tend-tini)/h)
t,V = integrador_ode_condicionado(rk4,fg,x0,tini,tend,k,p,g)  


# In[58]:


plt.plot(t,V[0,:])
plt.xlabel('t[ms]')
plt.ylabel('V[mV]')
plt.title('Gráfico donde I= I_0*cos(t/30)')
plt.legend()
plt.show()


# ### Apartado H

# In[6]:


i = np.linspace(0,5,500)
freq = np.piecewise(i,[i <= 1.5,i > 1.5], [0, lambda i:(1/(-tau*np.log(1-(V_u-E)/(R*i))))])


# In[7]:


# Gráfica de la función en rama
fig6 = plt.figure()
#plt.rcParamns["figure.figsize"]=[7,5] # Dimensiones de la figura
#plt.rcParamns['figure.dpi'] = 200 # Resolución
plt.title("Frecuencia de disparo en función de la corriente")
plt.xlabel("I")
plt.ylabel("frecuencia [KHz]")
plt.grid(1,linestyle='--')

plt.plot(i,freq,color='g',linewidth=2,label='frecuencia')
#plt.legend()
plt.show()


# In[84]:


# Graficos
fig4 = plt.figure()

plt.rcParams["figure.figsize"] = (7,5) # Dimensiones de la figura
plt.rcParams['figure.dpi'] = 200 # Resolución
plt.title('Gráfico de la frecuencia de disparo en función de la corriente')
plt.xlabel ("I")
plt.ylabel ("f[Hz]")
plt.grid(1,linestyle='--')

# Gráfico de función 

plt.plot(i, f, color = 'k', linewidth=2, label = 'frecuencia')
plt.legend ()
plt.show ()


# ### Apartado I

# In[71]:


def fih(x,t,p):   
    """
    x[0]= V
    p[0]= tau
    p[1]= E
    p[2]= R
    p[3]= I: R --> R
    """
    return (p[1]-x[0]+p[2]*p[3](t))/p[0]


# In[85]:


def I(t):
    return 0.35*(np.cos(t/3)+np.sin(t/5)+np.cos(t/7)+np.sin(t/11)+np.cos(t/13))**2
E = -65 #mV
R = 10 #M.Ohms
#I = 2 # nA
V_u = -50 # mV
p = [tau,E,R,I,V_u]
tini = 0 # ms
tend = 200 #ms
x0 = [E]
h = 0.05 #ms
k = int((tend-tini)/h)
t,V = integrador_ode_condicionado(rk4,fih,x0,tini,tend,k,p,g)


# In[86]:


plt.plot(t,E*np.ones(len(t)),color='red',linestyle=':',label='E')
plt.plot(t,V_u*np.ones(len(t)),color='blue',linestyle=':',label='V_u')
plt.plot(t,V[0,:],color='green',label='V rk4 + disp.')
dV = 0.1275
plt.xlabel('t[ms]')
plt.ylabel('V[mV]')
plt.title('Gráfico donde I_e(t) es igual a la suma de senos y cosenos')
plt.legend()
plt.show()


# In[ ]:




