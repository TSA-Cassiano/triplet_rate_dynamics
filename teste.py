import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
#plt.rcParams['figure.dpi'] = 200

coulomb = 1.60217662E-19
#plt.rcParams['figure.figsize'] = [8, 4]
experimental = np.loadtxt('experimental_PT0EP-CBP_8.txt') 
#experimental = experimental[1:]
y_exp    = experimental[:,1]/100 #experimental[:,1][0]
x_exp    = experimental[:,0]



tau_t    = 65*1E-6                    # s
q        = 1*coulomb                  # electron charge
#k_tt     = 3E-14                      # cm^3/s
d        = 1e-8  #0.8*1E-3*(k_tt*tau_t**2)/(4*q)                     # thickness of the exciton formation zone cm
kappa_tt = 2.5E-32                    # cm^6/s
J0       = 4 #1E3*4*q*d/(k_tt*tau_t**2)  # mA/cm^2
J        = np.linspace(x_exp[0],x_exp[-1],2000) # Current density mA/cm^2

norm = 0.028

##### Analytic dexter tta ####
def tta_dexter_analytic(J):
    return 0.025*(J0/(4*J))*(np.sqrt(1+8*(J/J0)) -1)
############################

##### Traditional Model ####
def tta_dexter(j): 
    # aT^2 + b*T + G*c = 0
    j = j*1E-3 #mA to A
    a = 1
    b = (2/(k_tt*tau_t))
    c = -(2/k_tt)
    G = j/(q*d)
    T = [x.real for x in np.roots([a,b,G*c]) if x.imag == 0 and x.real >=0][0]
    EFF = T/(G*tau_t)
    return EFF
############################


##### New Model ####
def tta_forster(j,param): 
    # aT^3 + 0*T^2 + b*T + G*c = 0
    j = j*1E-3 #mA to A
    a = -0.5*param*(norm)
    b = -1/tau_t
    c =  1
    G = j/(q*d)
    T = [x.real for x in np.roots([a,0,b,G*c]) if x.imag == 0 and x.real >=0][0]
    EFF = norm*T/(G*tau_t)
    return EFF
############################

fig, ax = plt.subplots(1,1)

#fitting 1 param only
def fitting(param_arr,x_exp,y_exp,func):
    mse_ar = []
    for param in param_arr:
        try_y = np.array([func(x,param) for x in x_exp])
        #try_y = try_y/try_y[0]
        MSE   = np.mean(np.square(y_exp - try_y))
        mse_ar.append(MSE)
        #ax.plot(x_exp,100*try_y,'--k') #,label=str(param),lw=2)
    indx = np.argmin(mse_ar)
    return param_arr[indx], mse_ar[indx]


#fitting 2 parameters
def sfitting(p1_arr,p2_arr,x_exp,y_exp,func):
    mse_ar = []
    for p2 in p2_arr:
        for p1 in p1_arr:
            try_y = np.array([func(x,p1) for x in x_exp])
            try_y = p2*try_y
            MSE   = np.mean(np.square(y_exp - try_y))
            #noise 
            #noise = [x*((-1)**(random.randint(0, 1)))*np.random.normal(1,1E-5,1) for x in y_exp]
            #MSE   = np.mean(np.square(y_exp - try_y +noise ))
            mse_ar.append([MSE,p1,p2])
            #ax.plot(x_exp,100*try_y,'--k') #,label=str(param),lw=2)
    indx = np.argmin(np.array(mse_ar)[:,0])
    return mse_ar[indx][0], mse_ar[indx][1],mse_ar[indx][2]



#rms,best_param_1,best_param_2 = fitting(np.linspace(1E-27,41E-27,100),np.linspace(0.1,1,10),x_exp,y_exp,tta_forster)
#print(rms,best_param_1,best_param_2)

best_param_1, mse = fitting(10**np.linspace(-39,-32,100),x_exp,y_exp,tta_forster)   #Com 1e-6: 1E-19,4E-19 #Sem 1e-6: 1E-26,4E-25 #Normalizado 1E-32,4E-32 -39,-32
print(mse,best_param_1)
#best_param_1 = 6.2802914418342725e-34
print('Radius:', 1e8*(best_param_1*tau_t/(4*np.pi))**(1/6))

analytic = tta_dexter_analytic(J)
#dexter   = np.array([tta_dexter(j) for j in J])
forster  = np.array([tta_forster(j,best_param_1) for j in J])   #best_param_2*np.array([tta_forster(j,best_param_1) for j in J])

#normalizing the data
#analytic = analytic/analytic[0]
#dexter   = dexter/dexter[0]
#forster  = forster/forster[0]
#analytic = analytic/analytic[0]

ax.plot(J,100*analytic,label='analytic',lw=3)
#ax.plot(J,100*dexter  ,label='dexter'  ,lw=3)
ax.plot(J,100*forster ,label='forster' ,lw=3) # /forster[0]
ax.scatter(x_exp,100*y_exp ,label='experimental',c='r',zorder=10)

ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_xlim([x0, 150])
#ax.set_ylim([10, 200])
ax.set_xlabel('Current Density (mA cm$^{-2}$)')
ax.set_ylabel('Normalized EQE (%)')
ax.legend()
plt.tight_layout()
plt.show()