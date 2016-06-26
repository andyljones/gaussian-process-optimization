import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sq_exp_kernel(p, q, length=.1, mag=2):
    return mag*sp.exp(-(sp.spatial.distance.cdist(p, q)/length)**2)

def evaluate(x_test, x_obs, y_obs, k=sq_exp_kernel, s=1e-3):
    koo = k(x_obs, x_obs) + s**2*sp.eye(len(x_obs))
    kto = k(x_obs, x_test)
    ktt = k(x_test, x_test)
    
    cho_koo = sp.linalg.cho_factor(koo)
    
    mu_test = kto.T.dot(sp.linalg.cho_solve(cho_koo, y_obs))
    sigma_test = ktt - kto.T.dot(sp.linalg.cho_solve(cho_koo, kto))
    
    return mu_test, sigma_test
    
def expected_improvement(x_test, x_obs, y_obs, **kwargs):
    mu_test, sigma_test = evaluate(x_test, x_obs, y_obs, **kwargs)
    
    best = y_obs.min()
    z_test = (best - mu_test)/sp.diag(sigma_test)[:, None]
    
    pdf = sp.stats.norm.pdf(z_test)
    cdf = sp.stats.norm.cdf(z_test)
    
    ei = (best - mu_test)*cdf + sp.diag(sigma_test)[:, None]*pdf

    return ei.clip(0, None)
    
def optimize(f, left, right, tol=1e-3, n=200, **kwargs):
    x_obs = sp.array([[(left + right)/2.]])
    y_obs = f(x_obs)
    x_test = sp.linspace(left, right, n)[:, None]
    
    ei = sp.array([[sp.inf]])
    while ei.max() > tol:
        ei = expected_improvement(x_test, x_obs, y_obs, **kwargs)
        print ei.max()
        x_new = x_test[sp.argmax(ei)]
        x_obs = sp.vstack([x_obs, [x_new]])
        y_obs = sp.vstack([y_obs, f(x_new)])
        
    return x_obs, y_obs
        
        
def gramacy_lee(x):
    return sp.sin(10*sp.pi*x)/(2*x) + (x - 1)**4