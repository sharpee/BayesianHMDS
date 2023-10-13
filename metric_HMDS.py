import numpy as np
import pystan as stan
import scipy.integrate as integrate
from scipy.special import gamma

# STAN code for finding the hyperbolic embedding in Lorentzian Coordinates
HMDS_code = """
functions {
    //hyperbolic distance between points in lorentz coordinates
    real hyp(real t1, real t2, vector E1, vector E2){
        real xi = t1*t2 - dot_product(E1, E2);
        if (xi <= 1)
            return 0.0;
        else
            return acosh(xi);
    }
}
data {
    int<lower=0> N;        // number of points
    int<lower=0> D;        // Dimension of space
    matrix[N, N] deltaij;  // matrix of data distances, only upper triangle will be operated on
}
transformed data {
    real Nterms = 0.5*N*(N-1);
}
parameters {
    vector[D] euc[N];               //x1-xd coordinates in lorentz space
    vector<lower=0.0>[N] sig;       //embedding uncertainties
    real<lower=0.001> lambda;       //scale parameter
}
transformed parameters {
    vector[N] time;      //xo coordinates in lorentz space, solve for from constraint eqs
    
    for (i in 1:N)
        time[i] = sqrt(1.0 + dot_self(euc[i]));
}
model {
    real dist;
    real seff;
    
    //normal prior on scale parameter
    target += -Nterms*0.5*square(lambda)/square(10.0);
      
    //inverse gamma priors on uncertainties
    for (i in 1:N)
        sig[i] ~ inv_gamma(2.0, 0.5);
    
    //loop over all pairs
    for(i in 1:N){
        for (j in i+1:N){
            if (deltaij[i,j] > 0.0){
                //compute embedding distance and effective uncertainties
                dist = hyp(time[i], time[j], euc[i], euc[j]);
                seff = sqrt(square(sig[i]) + square(sig[j]));

                //data distances normally distributed about embedding distance
                deltaij[i,j] ~ normal(dist/lambda, seff);
            }
        }
    }
}
"""

# STAN code for finding the center of mass of a collection of points in Lorentzian
# Coordinates
CM_code = """
functions {
    real hyp(real t1, real t2, vector E1, vector E2){
        return acosh(t1*t2 - dot_product(E1, E2));
    }
}
data {
    int<lower=0> N;        // number of points
    int<lower=0> D;        // dimension of space
    vector[D] coords[N];   // x1-xd lorentzian coordinates of data
}
transformed data{
    vector[N] coord_ts;
    
    for(i in 1:N)
        coord_ts[i] = sqrt(1.0 + dot_self(coords[i]));
}
parameters {
    vector[D] CM;                // center of mass x1-xd lorentzian coordinates
}
transformed parameters {
    real CM_t;
    
    CM_t = sqrt(1.0 + dot_self(CM));
}
model {
    real dist;
   
    for(i in 1:N){
        dist = hyp(CM_t, coord_ts[i], CM, coords[i]);

        target += -square(dist);
    }
}
"""

# postprocessing functions
###############################################################################################

def poincare_distance(p1, p2):
   """
   Return Hyperbolic distance between poincare coordinates of 2 points
   """
   inv = 2.0*(p1-p2).dot(p1-p2)/((1-p1.dot(p1))*(1-p2.dot(p2)))
   return np.arccosh(1.0 + inv)

def get_poincare_dmat(pts):
   """
   Return NxN distance matrix from a collection of points in their poincare coordinates
   """
   N = pts.shape[0]
   dm = np.zeros(shape=(N,N))

   for i in np.arange(N):
      for j in np.arange(i+1, N):
         dm[i][j] = poincare_distance(pts[i], pts[j])
         dm[j][i] = dm[i][j]

   return dm

def lorentz_to_poincare(pts):
   """
   Convert from lorentz to poincare coordinates. Only need to provide x1-xd lorentz coordinates
   """
   ts = np.sqrt(1.0 + np.sum(np.square(pts), axis=1))
   p_coords = (pts.T / (ts + 1)).T
   return p_coords

def poincare_to_lorentz(pts):
   """
   Convert from poincare to lorentz coordinates. Return both time and space components
   """
   x1_xd = 2.0*pts/(1.0 - np.sum(np.square(pts)))
   x0 = (1.0 + np.sum(np.square(pts)))/(1.0 - np.sum(np.square(pts)))
   return (x0, x1_xd)

def poincare_translation(v, x):
   """
   Isometric translation of point x in poincare coordinates.
   Hyperbolic translations are defined by how the origin is translated. Returns the
   translated position of x if the origin were to be translated to v.
   """
   dp = v.dot(x)
   v2 = v.dot(v)
   x2 = x.dot(x)
   
   return ((1.0 + 2.0*dp + x2)*v + (1.0 - v2)*x) / (1.0 + 2.0*dp + x2*v2)

def re_center(p_coords, CM_pc):
   """
   given poincare coordinates and ther center of mass, return the isometric translation of
   points so that the CM is now at the origin
   """
   return np.asarray([poincare_translation(-CM_pc, pt) for pt in p_coords])

def process_simulation(fit):
   """
   Compute all post-processing and add values to the optimizer dictionary. Operates in
   place, so returns nothing.
   Quantities added to dictionary:
      'poin' - poincare coordinates of embedding
      'rs' - radial coordinates of points
      'emb_mat' - hyperbolic distance matrix of embedded points
      'cp' - poincare coordinates of embedding with center of mass translated to the origin
      'crs' - radial coordinates of points after CM translation
   """
   fit['poin'] = lorentz_to_poincare(fit['euc'])
   fit['rs'] = 2.0*np.arctanh(np.sqrt(np.sum(np.square(fit['poin']), axis=1)))
   fit['emb_mat'] = get_poincare_dmat(fit['poin'])

   #compute center of mass of embedding, and translate points
   CM_m = stan.StanModel(model_code=CM_code, verbose=False)
   N, D = fit['poin'].shape
   CM_dat = {'N':N, 'D':D, 'coords':fit['euc']}
   CM_fit = CM_m.optimizing(data=CM_dat)
   CM_fit['poin'] = CM_fit['CM']/(1.0 + CM_fit['CM_t'])
   fit['cp'] = re_center(fit['poin'], CM_fit['poin'])
   fit['crs'] = 2.0*np.arctanh(np.sqrt(np.sum(np.square(fit['cp']), axis=1)))

###############################################################################################


# N: number of points, must be same as shape of dij
# D: Dimension of embedding space
# dij: distance matrix to be embedded
# iv: initial values to start embedding from
def embed(D, dij, initial_values=None, Niter=50000):
   """
   Returns embedding of distance matrix and optimized parameter values
   Inputs:
      D - Dimension of embedding space
      dij - dissimilarity matrix which embedding tries to reproduce
      initial_values (optional) - initial conditions to begin optimizing from
      Niter (default 50000) - max number of iterations optimizer runs for if convergence
         not detected
   Output:
      Dictionary containing processed embedding results stored in the following keys
      'euc' - x1-xd space like euclidean components of lorentzian representation of embedding
      'time' - x0 time like component of lorentz representation
      'lambda' - curvature scale parameter
      'sig' - pointwise embedding uncertainties
      'poin' - poincare coordinates of embedding
      'rs' - radial coordinates of points
      'emb_mat' - hyperbolic distance matrix of embedded points
      'dmat' - data dissimilarity matrix given as input
      'cp' - poincare coordinates of embedding with center of mass translated to the origin
      'crs' - radial coordinates of points after CM translation
   """

   #compile model
   hmds_m = stan.StanModel(model_code=HMDS_code, verbose=False)

   N = len(dij)
   dat = {'N':N, 'D':D, 'deltaij':dij}
   if (initial_values != None):
      fit = hmds_m.optimizing(data=dat, iter=Niter, init=initial_values)
   else:
      fit = hmds_m.optimizing(data=dat, iter=Niter)
   fit['dmat'] = dij
   process_simulation(fit)
   return fit

# functions for generating data uniformly distributed in hyperbolic space
###############################################################################################

# Compute volume of hyperbolic sphere of radius R in D dimensions
def vol(R, D):
   c = 2*np.power(np.pi, 0.5*D)
   itg = integrate.quad(lambda x: np.power(np.sinh(x), D-1), 0.0, R)[0]
   return c*itg

# sample radial coordinate of point uniformly out to max rad R
def sample_r(R, D):
   rmin = 0.0; rmax = R; rc = 0.5*R
   V_tot = vol(R, D); uc = vol(rc, D)/V_tot
   u = np.random.uniform()
   while(np.abs(u-uc)>1e-4):
       if uc < u:
           rmin = rc
           rc = 0.5*(rc + rmax)
           uc = vol(rc, D)/V_tot
       elif u < uc:
           rmax = rc
           rc = 0.5*(rc + rmin)
           uc = vol(rc, D)/V_tot
   return rc

# uniformly sample a point in hyperbolic space out to Rmax
# return point in poincare coords
def h_samp(rm, D, N):
   dirs = np.random.normal(size=(N,D))
   dirs = dirs/np.sqrt(np.sum(np.square(dirs), axis=1)).reshape(-1,1)
   r_p = np.asarray([sample_r(rm, D) for i in np.arange(N)])
   return np.tanh(r_p/2.0).reshape(-1,1)*dirs

# large scale embedding methods
###############################################################################################

#uncoupled HMDS model for initializing new pts around a seed
init_code = """
functions {
    real hyp(real t1, real t2, vector E1, vector E2){
        return acosh(t1*t2 - dot_product(E1, E2));
    }
}
data {
    int<lower=0> Ne;           // number of already embedded points
    int<lower=0> Nn;           // number of new points
    int<lower=0> D;            // Dimension of space
    real<lower=0.001> lambda;  // curvature of initial embedding
    vector[D] euc_emb[Ne];     // spacelike lorentzian coordinates of existing embedded points
    matrix[Nn, Ne] deltaij;    // matrix of data distances, only upper triangle will be operated on
}
parameters {
    vector[D] euc_new[Nn];                // directions
}
transformed parameters {
    vector[Ne] time_e;
    vector[Nn] time_n;
    
    for (i in 1:Ne)
        time_e[i] = sqrt(1.0 + dot_self(euc_emb[i]));
    for (i in 1:Nn)
        time_n[i] = sqrt(1.0 + dot_self(euc_new[i]));
}
model {
    real dist;
    
    for(i in 1:Nn){
        for (j in 1:Ne){
            if (deltaij[i,j] > 0.0){
                dist = hyp(time_n[i], time_e[j], euc_new[i], euc_emb[j]);

                deltaij[i,j] ~ normal(dist/lambda, 1.0);
            }
        }
    }
}
"""

def fit_seed(dmat, dim, Nseed, optimizer):
   red_dmat = dmat[:Nseed].T[:Nseed].T
   red_dmat = 2.0*red_dmat/np.max(red_dmat)

   dat = {'N':Nseed, 'D':dim, 'deltaij':red_dmat}
   seed_fit = optimizer.optimizing(data=dat, iter=100000)

   return(seed_fit)

def initialize_layer(sfit, dmat, Nadd, optimizer):
   N,D = sfit['euc'].shape
   tmp_max = np.max(dmat[:N].T[:N].T)
   init_dmat = 2.0*dmat[N:(N+Nadd)].T[:N].T / tmp_max

   init_dat = {'Ne':N, 'Nn':Nadd, 'D':D, 'lambda':sfit['lambda'],
               'euc_emb':sfit['euc'], 'deltaij':init_dmat}
   init_fit = optimizer.optimizing(data=init_dat, iter=50000, tol_rel_grad=1e2)
   init_fit['euc_emb'] = sfit['euc']

   sfit['euc'] = np.concatenate([init_fit['euc_emb'], init_fit['euc_new']], axis=0)


def relax_seed(sfit, dmat, optimizer):
   N, D = sfit['euc'].shape
   red_dmat = dmat[:N].T[:N].T
   red_dmat = 2.0*red_dmat/np.max(red_dmat)
   
   init = {'euc':sfit['euc'], 'lambda':sfit['lambda']}
   dat = {'N':N, 'D':D, 'deltaij':red_dmat}
   new_fit = optimizer.optimizing(data=dat, init = init, iter=500000)
   return new_fit


def large_embedding(Nseed, Nadd, Ntot, D, dij):
   #compile models
   hmds_m = stan.StanModel(model_code=HMDS_code, verbose=False)
   init_m = stan.StanModel(model_code=init_code, verbose=False)
   #embed seed distribution
   seed = fit_seed(dij, D, Nseed, hmds_m)

   #number of batches rounds down
   Nbatches = int(np.floor((Ntot-Nseed)/Nadd))

   #add more points in batches of Nadd
   for i in np.arange(Nbatches):
      print('adding %d th iteration of new points' % i)
      #initialize Nadd new points, passively update euc coords in seed
      initialize_layer(seed, dij, Nadd, init_m)
      #take seed with new points initialized and let all points relax freely
      seed = relax_seed(seed, dij, hmds_m)

   return seed

