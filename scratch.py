import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def k(p, q, length=.1, mag=1):
    return mag*sp.exp(-(sp.spatial.distance.cdist(p, q)/length)**2)

def evaluate(x_test, x_obs, y_obs, s=1e-3, **kwargs):
    ktt = k(x_test, x_test, **kwargs)        
    if len(x_obs) == 0:
        return sp.zeros(len(x_test)), ktt
    if x_obs.ndim == 1:
        x_obs = x_obs[:, None]
    
    koo = k(x_obs, x_obs, **kwargs) + s**2*sp.eye(len(x_obs))
    kto = k(x_obs, x_test, **kwargs)
    
    cho_koo = sp.linalg.cho_factor(koo)
    
    mu_test = kto.T.dot(sp.linalg.cho_solve(cho_koo, y_obs))
    sigma_test = ktt - kto.T.dot(sp.linalg.cho_solve(cho_koo, kto))    
    
    return mu_test, sigma_test
    
def expected_improvement(x_test, x_obs, y_obs, **kwargs):
    if len(x_obs) == 0:
        return sp.inf*sp.ones_like(x_test)
    
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
        
    return x_obs, y_obs.flatten()
    
### PRESENTATION FNS

def plot_optimization_step(x_obs, y_obs, x_test, i, **kwargs):
    plot_gp(x_obs[:i], y_obs[:i], x_test, **kwargs)
    if i < len(x_obs):
        plt.axvline(x_obs[i], c='r', alpha=0.3)

def plot_expected_improvement(x_obs, y_obs, x_test, i):
    ei = expected_improvement(x_test, x_obs[:i], y_obs[:i])
    plt.plot(x_test, ei)
    if i < len(x_obs):
        plt.axvline(x_obs[i], c='r', alpha=0.3)

def gramacy_lee(x):
    return 2*sp.sin(sp.pi*x/5) + 0.4*sp.cos(4*sp.pi*x/5) + 0.5*x
    
def plot_points(x_obs, y_obs):
    plt.scatter(x_obs, y_obs, s=100, alpha=0.5)
    
def plot_gp(x_obs, y_obs, x_test, **kwargs):
    mu_test, sigma_test = evaluate(x_test, x_obs, y_obs, **kwargs)
    diag = sp.sqrt(sp.diag(sigma_test)) + 1e-3
    
    plt.plot(x_test, mu_test)
    plt.fill_between(x_test.flatten(), (mu_test - 2*diag).flatten(), (mu_test + 2*diag).flatten(), alpha=0.3)
    
    plot_points(x_obs, y_obs)
    
def plot_config():
    plt.gcf().set_size_inches(12, 12)
    plt.xlim(-5, +5)    
    plt.ylim(-5, +5)
    
def sample(x_obs, y_obs, x_test, n=3, **kwargs):
    mu_test, sigma_test = evaluate(x_test, x_obs, y_obs, **kwargs)
    
    samples = sp.random.multivariate_normal(mu_test, sigma_test + 1e-6*sp.eye(len(sigma_test)), size=n)

    return samples
    
def plot_samples(x_test, samples):
    for s in samples:
        plt.plot(x_test, s)
    
def plot_covariance(x_obs, y_obs, x_test, **kwargs):
    mu_test, sigma_test = evaluate(x_test, x_obs, y_obs, **kwargs)
    
    with sns.axes_style('whitegrid'):
        extent = (x_test.min(), x_test.max(), x_test.min(), x_test.max())
        plt.imshow(sigma_test, interpolation='none', cmap=plt.cm.RdBu, vmin=-1, vmax=+1, extent=extent)    
        plt.gcf().set_size_inches(12, 12)
        plt.colorbar(fraction=.05)
    
#x_test = sp.linspace(-5, +5, 500)[:, None]
#kwargs = dict(length=1, mag=3)
#x_obs, y_obs = optimize(gramacy_lee, -5, +5, **kwargs) 
#plot_optimization_step(x_obs, y_obs, x_test, i=14, **kwargs)
#plt.plot(x_test, gramacy_lee(x_test))
#plot_config()
    
#x_obs = sp.array([0, 2, 2.1])
#y_obs = sp.array([1, 3, 4])
#x_test = sp.linspace(-5, +5, 500)[:, None]
#kwargs = dict(length=2, s=.5)
#
#plot_gp(x_obs, y_obs, x_test, **kwargs)
#plot_samples(x_test, sample(x_obs, y_obs, x_test, 3, **kwargs))
#plot_config()