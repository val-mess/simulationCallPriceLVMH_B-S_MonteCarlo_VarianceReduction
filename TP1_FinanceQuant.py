#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:11:11 2023

@author: valentinmessina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.stats import norm

#1-Simulation Monte Carlo
r= 0.045
T= 20
deltaT= 1 #1jour


# Chemin vers le fichier texte
chemin_fichier = 'LVMH_2023-11-23.txt'

# Lire le fichier en tant que DataFrame
df = pd.read_csv(chemin_fichier, delimiter='\t')

# Extraire les colonnes de volatilité
cours = df[['clot']]
Y = np.log(cours['clot'] / cours['clot'].shift(1))


#Traçage du cours

# Convertir la colonne 'clot' en un numpy array pour le traçage
closing_prices = cours['clot'].values

# Créer un axe temporel pour l'index du DataFrame
dates = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

# Traçage
# plt.figure(figsize=(10, 6))
# plt.plot(dates, closing_prices, linestyle='-', color='b')
# plt.title("Cours de l'action LVMH entre 23/11/2022 et 23/11/2023")
# plt.xlabel("Date")
# plt.ylabel("Cours à la fermeture")
# plt.grid(True)
# plt.show()


nu=0
for i in range(1,len(cours)):
    nu+=Y[i]

nu=nu/(deltaT*len(cours))


sigma2=0
for i in range(1,len(cours)):
    sigma2+=(Y[i]-nu*deltaT)**2

sigma2 = sigma2/(deltaT*(len(cours)-1))

sigma=np.sqrt(sigma2)

S0=cours.iloc[0]


def S(T):
    return S0[0]*np.exp((r-0.5*sigma2)*T+sigma*np.sqrt(T)*stats.norm.rvs(loc=0,scale=1, size=1)[0])

    
def prixCallMC(K,N):
    C=[]
    prix=[]
    for i in range(N):
        C.append(np.maximum(0,S(T)-K)*np.exp(-r*T))
        prix.append(np.mean(C))        
    return prix

#2-Prix B&S

def black_scholes_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price[0]

K=700
N=1000

callBlackScholes=black_scholes_call(S0, K, r, T, sigma)

# plt.figure()
# plt.plot(np.arange(N),prixCallMC(K,N),label='Prix du Call Monte-Carlo')
# plt.axhline(y=callBlackScholes, color='red', linestyle='--', label='Prix du Call B&S')
# plt.legend()
# plt.title("Comparaison prix Monte-Carlo et prix B&S")
# plt.xlabel("Nombre de simulations N")
# plt.ylabel("Prix du Call")

#Ajouter les intervalles de confiance
#3-Réduction de variance

def prixCallMC_VarContr(K, N):
    C_vc = []
    prix_vc = []
    C0 = prixCallMC(K, N)[N-1]
    betaL=[]
    beta=0

    for i in range(N):
        ST = S(T)
        varianceST = S0*S0*np.exp(2*r*T)*(np.exp(sigma2*T)-1)
        betaL.append((ST - S0 * np.exp(r * T)) * (np.exp(-r * T) *np.maximum(0, ST - K) - C0 )  )                
        beta=np.mean(betaL)/varianceST
        
    for i in range(N):
        ST = S(T)
        C_vc.append( np.exp(-r * T)*np.maximum(0, ST - K)- beta*(ST-S0*np.exp(r * T)) )
        prix_vc.append(np.mean(C_vc))
    
    return prix_vc


#callBlackScholes=black_scholes_call(S0, K, r, T, sigma)

# plt.figure
# plt.plot(np.arange(N),prixCallMC_VarContr(K,N),label='Prix du Call Monte-Carlo Variable Contrôle')
# plt.plot(np.arange(N),prixCallMC(K,N),label='Prix du Call Monte-Carlo')
# plt.axhline(y=callBlackScholes, color='red', linestyle='--', label='Prix du Call B&S')
# plt.legend()
# plt.title("Comparaison prix Monte-Carlo Variable Contrôle et prix B&S")
# plt.xlabel("Nombre de simulations N")
# plt.ylabel("Prix du Call")




#test ajout intervalles de confiance

def prixCallMC_IC(K, N):
    C=[]
    prix=[]
    for i in range(N):
        C.append(np.maximum(0,S(T)-K)*np.exp(-r*T))
        prix.append(np.mean(C))        
        
    # Calculate mean and standard deviation for confidence intervals
    mean_prix = np.mean(prix)
    std_prix = np.std(prix, ddof=1)  # Use ddof=1 for sample standard deviation

    # Calculate confidence intervals
    confidence_interval = 1.96 * (std_prix / np.sqrt(N))

    return prix, confidence_interval

def prixCallMC_VarContr_IC(K, N):
    C_vc = []
    prix_vc = []
    C0 = prixCallMC(K, N)[N-1]
    betaL=[]
    beta=0

    for i in range(N):
        ST = S(T)
        varianceST = S0*S0*np.exp(2*r*T)*(np.exp(sigma2*T)-1)
        betaL.append((ST - S0 * np.exp(r * T)) * (np.exp(-r * T) *np.maximum(0, ST - K) - C0 )  )                
        beta=np.mean(betaL)/varianceST
        
    for i in range(N):
        ST = S(T)
        C_vc.append( np.exp(-r * T)*np.maximum(0, ST - K)- beta*(ST-S0*np.exp(r * T)) )
        prix_vc.append(np.mean(C_vc))

    # Calculate mean and standard deviation for confidence intervals
    mean_prix_vc = np.mean(prix_vc)
    std_prix_vc = np.std(prix_vc, ddof=1)  # Use ddof=1 for sample standard deviation

    # Calculate confidence intervals
    confidence_interval_vc = 1.96 * (std_prix_vc / np.sqrt(N))

    return prix_vc, confidence_interval_vc


prixCallMC_var_contr, conf_interval_VC = prixCallMC_VarContr_IC(K, N)
prixCallMC_, conf_interval_MC = prixCallMC_IC(K, N)

plt.figure()

plt.plot(np.arange(N), prixCallMC_, label='Prix du Call Monte-Carlo')
#plt.errorbar(np.arange(N), prixCallMC_, yerr=conf_interval_MC, label='IC à 5% MC', alpha=0.05, fmt='o')

plt.plot(np.arange(N), prixCallMC_var_contr, label='Prix du Call Monte-Carlo VC', color='black')
plt.axhline(y=callBlackScholes, color='red', linestyle='--', label='Prix du Call B&S')
plt.errorbar(np.arange(N), prixCallMC_var_contr, yerr=conf_interval_VC, label='IC à 5% MC-VC', alpha=0.05, fmt='o',color='grey')
plt.legend()
plt.title("Comparaison prix Monte-Carlo simple et VC avec prix B&S avec IC 5%")
plt.xlabel("Nombre de simulations N")
plt.ylabel("Prix du Call")
plt.show()

plt.figure()
#plt.errorbar(np.arange(N), prixCallMC_, yerr=conf_interval_MC, label='IC à 5% MC', alpha=0.05, fmt='o')
plt.plot(np.arange(N), prixCallMC_var_contr, label='Prix du Call Monte-Carlo VC', color='black')
plt.axhline(y=callBlackScholes, color='red', linestyle='--', label='Prix du Call B&S')
plt.errorbar(np.arange(N), prixCallMC_var_contr, yerr=conf_interval_VC, label='IC à 5% MC-VC', alpha=0.05, fmt='o',color='grey')
plt.legend()
plt.title("Comparaison prix Monte-Carlo VC prix B&S avec IC 5%")
plt.xlabel("Nombre de simulations N")
plt.ylabel("Prix du Call")
plt.show()