# Atmospheric thermodynamic constant and variables
# Olivier Pauluis - 2018
#
# Note: by default, the thermodynamics functions here are defined using mixing ratios 'r' 
# defined as the amount of water per unit mass of dry air. Switching to specific humidity 
# (i.e. mass of water per unit of total msaa is strightforward, with
# q = r / (1 + r_T)
#
# Pressure is in Pascal
# Energy in J/kg
# Entropy in J/kg/K

from numpy import log,exp,minimum, maximum
from numba import jit

# Heat capacity of dry air, water vapor, liquid water, and ice (J/Kg/K)

Cpd = 1007.
Cpvap = 1880
Cliq =4218
Cice = 2106

#Ideal gas constant
Rd = 287.04
Rvap = 461.5

Kappa_d= Rd/Cpd

#latent heat of vaporization and fusion at 0C
Lf0 = 333.7e3
Lv0 =2500.0e3

#Triple point under 1 bar
Tref = 273.15

#Reference ratio
Pref = 1.0e5

#For virtual temperature
Rrat = Rvap / Rd - 1.0



def es_water(T):
    '''Saturation vapor pressure over water at tempertaure T (in K)'''
    C1 = -9.09718
    C2 = -3.56654
    C3 =  0.876793 
    C4 = 2.78583503
    C5 = -7.90298 
    C6 =  5.02808 
    C7 = -1.3816E-7
    C8 = 0.0081328
    D1 = 11.344
    D2 = -3.49149
    F5 = 5.0057149
    ln10=log(10.0)

    tboil=373.15

    C  = tboil/T
    E1 = D1*(C - 1.0)/C * ln10
    E2 = D2*(C - 1.0)   * ln10
    F1 = C5*(C - 1.0)
    F3 = C7*(exp(E1) - 1.0)
    F4 = C8*(exp(E2) - 1.0)
    F2 = C6*log(C)/ln10
    B  = ln10*(F1 + F2 + F3 + F4 + F5)
    A  = 1.0
    
    ESAT = A*np.exp(B)
    return ESAT

def es_ice(T):
    '''Saturation vapor pressure over ice at temperature T(K)'''

    ln10=log(10.0)
    C1 = -9.09718
    C2 = -3.56654
    C3 = 0.876793
    C4 = 2.78583503

    TFREEZE = 273.15

    C  = TFREEZE/T
    F1 = C1*(C - 1.0)
    F3 = C3*(C - 1.0)/C
    F4 = C4
#    F2 = C2*(log(C,10.0)) 
    F2 = C2*log(C)/ln10
    B  = ln10*(F1 + F2 + F3 + F4)
    A  = 1.0
    ESAT = A*exp(B)

    return ESAT

def es_mix(T):
    '''stauration vapor over water (T>0C) or ice (T<0C)'''
    esw = es_water(T)
    esi = es_ice(T)
    esat = minimum(esi,esw)
    return esat

def vapor_pressure(P,rv):
    '''partial pressure of water vapor'''
    E = P* Rvap * rv /(Rd +   Rvap *rv)
    return E

def mixing_ratio(P,e):
    '''mixing ratio of water vapor'''
    rv = Rd * E /(P  - E )/   Rvap
    return rv

def sat_mixing_ratio(P,T):
    '''mixing ratio of water vapor'''
    ES = minimum(es_mix(T),P - 0.5)
#    ES = min(es_water(T),P -1.0e-10)    
    rv = Rd * ES /(P  - ES )/   Rvap
    return rv

def moist_entropy(P,T,rv,rl,ri):
    '''moist entropy as function of pressure, temperature, and mixing ration for water vapor, liquid water and ice'''

    esat=es_water(T)
    E = P* Rvap * rv /(Rd +   Rvap *rv)
    Pd = P - E
    CP = Cpd + (rv + rl) * Cliq + ri * Cice
    LV = Lv0 + (Cpvap - Cliq ) * (T - Tref)
    entropy =  CP * log(T/Tref) - Rd *  log(Pd/Pref) - rv * Rvap *log(E/esat) +  rv*LV/T - Lf0*ri/ Tref
    return entropy

def ice_entropy(P,T,rv,rl,ri):
    '''entropy over ice as function of pressure, temperature, and mixing ration for water vapor, liquid water and ice'''

    esat=es_water(T)
    E = P * Rvap * rv /(Rd +   Rvap *rv)
    Pd = P - E
    CP = Cpd + (rv + rl) * Cliq + ri * Cice
    LV = Lv0 + (Cpvap - Cliq ) * (T - Tref)

    S =  CP * log (T/Tref) - Rd *  log(Pd/Pref) - rv * Rvap *log(E/esat) +  rv*Lv/T + Lf0*(rv+rl)/ Tref
    return S

def equivalent_potential_temperature(P,T,rv,rl,ri):
    ''' Equivalent potential temperature, based on Emanuel (1994), but includes ice phase contribution.'''

    S = moist_entropy(P,T,rv, rl, ri)
    rt = rv + rl + ri
    CP =  Cpd + rt * Cliq
    THETA_E= Tref * exp(S/CP)
    return THETA_E

def frozen_equivalent_potential_temperature(P,T,rv,rl,ri):
    ''' Frozen equivalent potential temperature, based on Pauluis (2018).'''

    S = moist_entropy_ice(P,T,rv, rl, ri)
    rt = rv + rl + ri
    CP =  Cpd + rt * Cice
    THETA_E= Tref * exp(S/CP)
    return THETA_E

def Gibbs_vapor(P,T,rv):
    '''Gibbs free energy for water vapor'''
    ES=es_water(T)
    E = P* Rvap * rv /(Rd +   Rvap *rv)
    GV = Cliq* ( T-Tref - T*(log(T/Tref))) + Rvap * T * log(E/ES)
    return GV

def Gibbs_liquid(T):
    '''Gibbs free energy for liquid water'''
    GL = Cliq* ( T-Tref - T*(log(T/Tref))) 
    return GL

def Gibbs_water(T):
    '''Gibbs free energy for liquid water'''
    GL = Cliq* ( T-Tref - T*(log(T/Tref))) 
    return GL

def Gibbs_ice(T):
    '''Gibbs free energy for liquid water'''
    GI = Cice* ( T-Tref - T*(log(T/Tref))) + Lf0 * (T/Tref - 1.0) 
    return GI

def saturated_moist_entropy(P,T,rt):
    '''saturated moist entropy as function of pressure, temperature, and total water'''
    esat=es_water(T)
    rv = minimum(mixing_ratio(P,esat),rt)
    E = vapor_pressure(P,rv)
    Pd = P - E
    rl = rt - rv
    CP = Cpd + (rv + rl) * Cliq 
    LV = Lv0 + (Cpvap - Cliq ) * (T - Tref)
    entropy =  CP * log(T/Tref) - Rd *  log(Pd/Pref) - rv * Rvap *log(E/esat) +  rv*LV/T 
    return entropy,rv,rl

def saturated_moist_entropy_ice(P,T,rt):
    '''saturated moist entropy as function of pressure, temperature, and total water, assuming condensed water is in ice form'''
    esat=es_ice(T)
    rv = minimum(mixing_ratio(P,esat),rt)
    E = vapor_pressure(P,rv)
    Pd = P - E
    ri = rt - rv
    CP = Cpd + rv  * Cliq + ri * Cice 
    LV = Lv0 + (Cpvap - Cliq ) * (T - Tref)
    entropy =  CP * log(T/Tref) - Rd *  log(Pd/Pref) - rv * Rvap *log(E/esat) +  rv*LV/T - ri * Lf0/Tref 
    return entropy,rv,ri

def saturated_moist_entropy_triple(P,rt,rl):
    '''saturated moist entropy as function of pressure, total water and ice water at the triple point '''
    esat=es_water(Tref)
    rv = minimum(mixing_ratio(P,esat),rt)
    E = vapor_pressure(P,rv)
    Pd = P - E
    rlt = minimum(rt-rv,rl)
    ri = max(rt - rlt,0.)
    LV = Lv0 
    entropy =   - Rd *  log(Pd/Pref) - rv * Rvap *log(E/esat) +  rv*LV/Tref - ri * Lf0/Tref 
    return entropy,rv,ri

def alpha_dry(P,T,rv):
    '''specific volume per unit mass of dry air as function of pressure, temperature and mixing ratio'''
    alpha = (Rd + Rvap * rv) * T / P
    return alpha

def density(p,T,rv,rl,ri):
    '''density  as function of pressure, temperature and mixing ratio'''
    rho = p / (Rd + Rvap * rv)/ T * (1.0 +  rv + rl + ri)
    return rho

def density_dry(p,T,rv,rl,ri):
    '''density  as function of pressure, temperature and mixing ratio'''
    rho_d = p / (Rd + Rvap * rv) / T
    return rho_d
