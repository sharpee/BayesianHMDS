import metric_HMDS as HMDS
import numpy as np

#sample 100 points uniformly in a 3D hyperbolic space out to radius 5
pts = HMDS.h_samp(5.0, 3, 100)
#compute and normalize their distance matrix
dmat = HMDS.get_poincare_dmat(pts)
dmat = 2.0*dmat/np.max(dmat)
print('made dmat')

#fit and process the model
fit = HMDS.embed(3, dmat)
#print the fitted scale parameter (should be close to 5 in this case)
print(fit['lambda'])
