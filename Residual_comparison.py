import os
import sys
import pathlib
#string libraries
import re
#arithmetic libraries
import numpy as np
import numpy.matlib
from scipy import interpolate as interp
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt

def legend_without_duplicate_labels(ax, a):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=a)

allprofs = pd.read_csv('/home/user/GlobalRegression/GlobalRegression/all_profiles_limited/all_profiles_limited_stan_residuals.csv')

stiff = pd.read_csv('/home/user/GlobalRegression/GlobalRegression/stiffer_profiles_limited/stiffer_profiles_limited_stan_residuals.csv')

soft = pd.read_csv('/home/user/GlobalRegression/GlobalRegression/softer_profiles_limited/softer_profiles_limited_new_stan_residuals.csv')

in_between = pd.read_csv('/home/user/GlobalRegression/GlobalRegression/in_between_profiles_limited/in_between_profiles_limited_stan_residuals.csv')


#total residuals versus depth
i_sort    = np.argsort( stiff.Depth_MPt.values )
x_data    = stiff.Depth_MPt[i_sort]

y_data_Jian   = stiff.res_orig[i_sort]
y_data_without_db   = allprofs.res_tot[i_sort]
y_data_my_function   = stiff.res_tot[i_sort]
y_data_with_db   = soft.res_tot[i_sort]
y_data_just_db   = in_between.res_tot[i_sort]

x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)

spl_mmean_Jian = interp.UnivariateSpline(x_data,y_data_Jian)
spl_mmean_without_db = interp.UnivariateSpline(x_data,y_data_without_db)
spl_mmean_my_function = interp.UnivariateSpline(x_data,y_data_my_function)
spl_mmean_with_db = interp.UnivariateSpline(x_data,y_data_with_db)
spl_mmean_just_db = interp.UnivariateSpline(x_data,y_data_just_db)

#spl_mmean.set_smoothing_factor(2)
y_mmean_Jian   = spl_mmean_Jian(x_mmean)
y_mmean_without_db   = spl_mmean_without_db(x_mmean)
y_mmean_my_function   = spl_mmean_my_function(x_mmean)
y_mmean_with_db   = spl_mmean_with_db(x_mmean)
y_mmean_just_db   = spl_mmean_just_db(x_mmean)

fname_fig = 'total_residuals_versus_depth'
fig, ax = plt.subplots(figsize = (10,10))

hl0 = ax.plot(y_data_Jian, x_data, 'o', markersize=4, label='Shi and Asimaki, 2018')
hl1 = ax.plot(y_data_without_db, x_data, 'o', markersize=4, label='all profiles')
hl2 = ax.plot(y_data_my_function, x_data, 'o', markersize=4, label='stiffer profiles')
hl3 = ax.plot(y_data_with_db, x_data, 'o',  c='k',markersize=4, label='softer profiles')
hl4 = ax.plot(y_data_just_db, x_data, 'o',  c='r',markersize=4, label='limited-range profiles')

#edit properties
ax.set_xlabel('total residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

legend_without_duplicate_labels(ax, 20)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
#ax.set_title(r'Total Residuals versus Depth Comparison', fontsize=30)
ax.legend(loc='lower left', fontsize=20)
fig.tight_layout()

fig.savefig('/home/user/Downloads/Data/Residual_Comparison_cases2.png' )

## residual comparison of both approaches
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

df_velprofs = pd.read_csv('~/Downloads/Data/all_velocity_profiles_new.xlsx')
df_velprofs = df_velprofs.loc[~np.isnan(df_velprofs.Depth_MPt),:]
df_velprofs_limited = df_velprofs[np.logical_and(df_velprofs['Vs30'].values>=0, df_velprofs['Vs30'].values<=1000)]
df_velprofs.reset_index(drop=True, inplace=True)
df_velprofs_limited.reset_index(drop=True, inplace=True)

#velocity profile ids
vel_id_dsid1, vel_idx1, vel_inv1 = np.unique(df_velprofs[['DSID','VelID']].values, axis=0, return_index=True, return_inverse=True)
vel_ids1  = vel_inv1 + 1
n_vel1    = vel_idx1.shape[0]

df_velinfo1  = df_velprofs[['DSID', 'DSName','VelID','VelName','Vs30','Lat','Lon','Depth_MPt','Thk', 'Vs', 'flag_Z1']]

#original scaling coefficients
#Vs0 scaling
p1_orig = -2.1688*10**(-4)
p2_orig = 0.5182
p3_orig = 69.452
#k scaling
r1_orig =-59.67
r2_orig =-0.2722
r3_orig = 11.132
#n scaling
s1_orig = 4.110
s2_orig =-1.0521*10**(-4)
s3_orig =-10.827
s4_orig =-7.6187*10**(-3)
#k scaling
r1_new = 3.42271
r2_new = 0.000227199
r3_new = 8.932775
r4_new = 2.43589
#n scaling
s1_new = 7.27457
s2_new = 0.072124
s3_new = 4.005865

z_star=2.5

vs30_array = np.logspace(np.log10(10), np.log10(2000))
#approach1
param_k_new1   = 10**(2.395*(np.log10(df_velprofs.Vs30.values)-1.7518)-2)#r1_new1 * (df_velprofs.Vs30.values - r2_new1)**r3_new1
param_n_new1   = 10**(1.917*sigmoid((np.log10(df_velprofs.Vs30.values)-1.963)*5.967)-1.312)#np.exp(s1_new1+(s2_new1/(1+s3_new1*((df_velprofs.Vs30.values)**(-s4_new1)))));
param_a_new1   =-1/param_n_new1
param_vs0_new1 = (param_k_new1*(param_a_new1+1)*z_star + (1+param_k_new1*(30-z_star))**(param_a_new1+1) - 1) / (30*(param_a_new1+1)*param_k_new1) * df_velprofs.Vs30.values

#approach2
param_k_new   = np.exp((r1_new / (1+r2_new*((0.001*df_velprofs.Vs30.values)**-r3_new)))- r4_new);#np.exp( dB_r_new[vel_inv] )
param_n_new   = (s1_new/(1+s2_new*((0.001*df_velprofs.Vs30.values)**(-s3_new))))+1;
param_a_new   =-1/param_n_new
param_vs0_new = (param_k_new*(param_a_new+1)*z_star + (1+param_k_new*(30-z_star))**(param_a_new+1) - 1) / (30*(param_a_new+1)*param_k_new) * df_velprofs.Vs30.values

#orignal profile parameters
param_k_orig   = np.exp(r1_orig*((df_velprofs.Vs30.values)**r2_orig) + r3_orig)
param_n_orig   = s1_orig*np.exp(s2_orig*df_velprofs.Vs30.values) + s3_orig*np.exp(s4_orig*df_velprofs.Vs30.values)
param_vs0_orig = p1_orig*(df_velprofs.Vs30.values)**2 + p2_orig*df_velprofs.Vs30.values + p3_orig

#mean prediction
y_data  = np.log(df_velprofs.Vs.values)#stan_data['Y'] 
y_new1   = np.log(param_vs0_new1    * ( 1 + param_k_new1    * ( np.maximum(0,df_velprofs.Depth_MPt.values-z_star) ) )**(1/param_n_new1))
y_new2   = np.log(param_vs0_new    * ( 1 + param_k_new    * ( np.maximum(0, df_velprofs.Depth_MPt.values-z_star) ) )**(1/param_n_new))
y_orig  = np.log(param_vs0_orig   * ( 1 + param_k_orig   * ( np.maximum(0, df_velprofs.Depth_MPt.values-z_star) ) )**(1/param_n_orig))
    
#compute residuals
res_tot1     = y_data - y_new1
res_tot2     = y_data - y_new2
res_orig    = y_data - y_orig

#summary predictions and residuals
predict_summary = np.vstack((np.exp(y_data), res_tot1,res_tot2, res_orig,)).T
columns_names   = ['VsProf_mean','res_tot1','res_tot2','res_orig']
df_predict_summary = pd.DataFrame(predict_summary, columns = columns_names, index=df_velprofs.index)
#create dataframe with predictions and residuals
df_predict_summary = pd.merge(df_velinfo1, df_predict_summary, how='right', left_index=True, right_index=True)
df_predict_summary[['DSID','VelID']] = df_predict_summary[['DSID','VelID']].astype(int)
#df_predict_summary.to_csv(dir_out + fname_out_main + '_stan_residuals' + '.csv', index=False)

#%% Comparison
### ======================================
from scipy.interpolate import splev
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html

class Splinefunc:
    def __init__( self, knots, coefs, degree ):
        self.knots = knots
        self.coefs = coefs
        self.degree = degree

    def __call__( self, x ):
        return splev( x, (self.knots, self.coefs, self.degree ))
      
# Total Residual
# ---------------------------
#total residuals versus depth
i_sort    = np.argsort( df_predict_summary.Depth_MPt.values )
x_data1    = df_predict_summary.Depth_MPt[i_sort]
y_data1    = df_predict_summary.res_tot1[i_sort]
x_data2    = df_predict_summary.Depth_MPt[i_sort]
y_data2    = df_predict_summary.res_tot2[i_sort]
x_data3   = df_predict_summary.Depth_MPt[i_sort]
y_data3   = df_predict_summary.res_orig[i_sort]

x_mmean1   = np.linspace(x_data1.min(), x_data1.max(), 1000)
spl_mmean1 = interp.UnivariateSpline(x_data1,y_data1)
#spl_mmean.set_smoothing_factor(2)
y_mmean1   = spl_mmean1(x_mmean1)

x_mmean2   = np.linspace(x_data2.min(), x_data2.max(), 1000)
spl_mmean2 = interp.UnivariateSpline(x_data2,y_data2)
#spl_mmean.set_smoothing_factor(2)
y_mmean2   = spl_mmean2(x_mmean2)

x_mmean3   = np.linspace(x_data3.min(), x_data3.max(), 1000)
spl_mmean3 = interp.UnivariateSpline(x_data3,y_data3)
#spl_mmean.set_smoothing_factor(2)
y_mmean3   = spl_mmean3(x_mmean3)

#fname_fig = (fname_out_main + '_residual_comparison').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#hl = ax.plot(y_data1, x_data1, 'o', c='k', markersize=4, label='Approach 1')
hl = ax.plot(y_data2, x_data2, 'o', c='r', markersize=4, label='Proposed Model')
hl = ax.plot(y_data3, x_data3, 'o', c='yellowgreen', markersize=4, label='Shi and Asimaki, 2018')

#hl = ax.plot(y_mmean1, x_mmean1,  '-', c='g', linewidth=4, label='Approach 1-spline')
hl = ax.plot(y_mmean2, x_mmean2, '-', c='k', linewidth=3, label='Proposed Model-spline')
hl = ax.plot(y_mmean3, x_mmean3,  '-', c='b', linewidth=3, label='Shi and Asimaki, 2018-spline')

#edit properties
ax.set_xlabel('total residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
#ax.set_title(r'Total Residuals versus Depth', fontsize=30)
ax.legend(loc='lower left',fontsize=15)
#fig.tight_layout()
fig.savefig( '/home/user/Downloads/Data/Residual_Model_Jian.png' )


'''
Comparison of Prior and Posterior Distributions 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import gaussian_kde
fpath =r"/home/user/GlobalRegression/GlobalRegression/Final/"
posterior = pd.read_csv(fpath+"stiffer_profiles_limited_stan_posterior_raw.csv")

# Parameters for the log-normal prior distribution
mu = 1.92
sigma = 0.3
# Generate x-axis values
x = np.linspace(2, 14, 1000)
# Calculate the PDF of the log-normal distribution
prior_pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))

# Posterior distribution
post1 = posterior["s1"][0:15000]
post2 = posterior["s1"][15000:30000]
post3 = posterior["s1"][30000:45000]
post4 = posterior["s1"][45000:60000]
post5 = posterior["s1"][60000:75000]
post6 = posterior["s1"][75000:90000]
# Estimate the PDF of the posterior distribution using KDE
kde1 = gaussian_kde(post1)
kde2 = gaussian_kde(post2)
kde3 = gaussian_kde(post3)
kde4 = gaussian_kde(post4)
kde5 = gaussian_kde(post5)
kde6 = gaussian_kde(post6)
# Generate x-axis values
x1 = np.linspace(min(post1), max(post1), 1000)
x2 = np.linspace(min(post2), max(post2), 1000)
x3 = np.linspace(min(post3), max(post3), 1000)
x4 = np.linspace(min(post4), max(post4), 1000)
x5 = np.linspace(min(post5), max(post5), 1000)
x6 = np.linspace(min(post6), max(post6), 1000)
# Calculate the PDF of the posterior distribution
posterior_pdf1 = kde1.evaluate(x1)
posterior_pdf2 = kde2.evaluate(x2)
posterior_pdf3 = kde3.evaluate(x3)
posterior_pdf4 = kde4.evaluate(x4)
posterior_pdf5 = kde5.evaluate(x5)
posterior_pdf6 = kde6.evaluate(x6)


# Plotting the sorted posterior distribution
fig, ax = plt.subplots(figsize = (5,4))

h=plt.plot(x, prior_pdf, c='orangered', label='Prior')

h=plt.plot(x1, posterior_pdf1, c='b', linestyle='-.', alpha=0.5)
h=plt.plot(x2, posterior_pdf2, c='b', linestyle=':', alpha= 0.6)
h=plt.plot(x3, posterior_pdf3, c='b', linestyle='dotted', alpha= 0.7)
h=plt.plot(x4, posterior_pdf4, c='b', linestyle='--', alpha= 0.8)
h=plt.plot(x5, posterior_pdf5, c='b', linestyle='-.', alpha= 0.9)
h=plt.plot(x6, posterior_pdf6, c='b', linestyle='-', alpha= 1, label='Posterior')

# Edit
#ax.set_xlabel(r'$r_1$', fontsize=10)
ax.set_ylabel(r'Frequency',  fontsize=20)
#ax.legend(loc='upper left', fontsize=17)
ax.grid(which='both')
ax.set_xlim([min(x), max(x)])
#ax.set_ylim([10, 2000])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
fig.tight_layout()
fig.savefig( fpath + 'prior_posterior/s1.png' )





