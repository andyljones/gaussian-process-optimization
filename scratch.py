import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def k(p, q, length=.1, mag=2):
    return mag*sp.exp(-(sp.spatial.distance.cdist(p, q)/length)**2)
    
def gramacy_lee(x):
    return sp.sin(10*sp.pi*x)/(2*x) + (x - 1)**4

def evaluate(x_test, x_obs, y_obs):
    koo = k(x_obs, x_obs)
    kto = k(x_obs, x_test)
    ktt = k(x_test, x_test)
    
    cho_koo = sp.linalg.cho_factor(koo)
    
    mu_test = kto.T.dot(sp.linalg.cho_solve(cho_koo, y_obs))
    sigma_test = ktt - kto.T.dot(sp.linalg.cho_solve(cho_koo, kto))
    
    return mu_test, sigma_test
    
def expected_improvement(x_test, x_obs, y_obs):
    mu_test, sigma_test = evaluate(x_test, x_obs, y_obs)
    
    best = y_obs.min()
    z_test = (best - mu_test)/sp.diag(sigma_test)[:, None]
    
    pdf = sp.stats.norm.pdf(z_test)
    cdf = sp.stats.norm.cdf(z_test)
    
    ei = (best - mu_test)*cdf + sp.diag(sigma_test)[:, None]*pdf

    return ei.clip(0, None)