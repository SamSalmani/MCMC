#load variables
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
from matplotlib.ticker import  AutoLocator as plt_autotick
import arviz as az
#mpl.use('agg')

#stan library
import cmdstanpy

sys.path.insert(0,'../python_lib/plotting')
import pylib_contour_plots as pycplt
#stan library
#import cmdstanpy
###from cmdstanpy import cmdstan_path, CmdStanModel
###import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)

#user functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#%% Define Variables
### ======================================
#constants
z_star = 2.5

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

#regression info
fname_stan_model = '~/GlobalRegression/jian_fun_glob_reg_upd3.0_just_db_log_res.stan'
#iteration samples
n_iter_warmup   = 15000
n_iter_sampling = 15000
#MCMC parameters
n_chains        = 6
adapt_delta     = 0.8
max_treedepth   = 10


# Input/Output
# --------------------------------
#input flatfile
fname_flatfile = '~/Downloads/Data/all_velocity_profiles_new.xlsx'
# fname_flatfile = '../../Data/vel_profiles_dataset/Jian_velocity_profles.csv'

#flag truncate
flag_trunc_z1 = False
#flag_trunc_z1 = True

#output filename
fname_out_main = 'k_just_db2'
# fname_out_main = 'Jian'
###fname_out_main = 'all_trunc'
#output directory
###dir_out = '../../Data/global_reg/bayesian_fit/JianFunUpd3_log_res/' + fname_out_main + '/'
dir_out = './GlobalRegression/' + fname_out_main + '/'
dir_fig = dir_out + 'figures/'

#%% Load Data
### ======================================
#load velocity profiles
df_velprofs = pd.read_csv(fname_flatfile)

#remove columns with nan mid-depth
df_velprofs = df_velprofs.loc[~np.isnan(df_velprofs.Depth_MPt),:]

#truncate profiles at depth of 1000m/sec
###if flag_trunc_z1:
###    df_velprofs = df_velprofs.loc[~df_velprofs.flag_Z1,:]

#reset index
df_velprofs.reset_index(drop=True, inplace=True)

#%% Regression
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#unique datasets
###ds_ids, ds_idx = np.unique(df_velprofs.DSID, return_index=True)
###ds_names       = df_velprofs.DSName.iloc[ds_idx].values

#velocity profile ids
vel_id_dsid, vel_idx, vel_inv = np.unique(df_velprofs[['DSID','VelID']].values, axis=0, return_index=True, return_inverse=True)
vel_ids  = vel_inv + 1
n_vel    = vel_idx.shape[0]

# regression input data
# ---   ---   ---   ---
#prepare regression data
stan_data = {'N':       len(df_velprofs),
             'NVEL':    n_vel,
             'i_vel':   vel_ids,
             'z_star':  z_star,
             'Vs30':    df_velprofs.Vs30.values[vel_idx],
             'Z':       df_velprofs.Depth_MPt.values,
             'Y':       np.log(df_velprofs.Vs.values),
            }
#write as json file
fname_stan_data = dir_out +  'gobal_reg_stan_data' + '.json'
try:
    cmdstanpy.utils.jsondump(fname_stan_data, stan_data)
except AttributeError:
    cmdstanpy.utils.write_stan_json(fname_stan_data, stan_data)

# run stan
# ---   ---   ---   ---
#compile stan model
stan_model = cmdstanpy.CmdStanModel(stan_file=fname_stan_model) 
stan_model.compile(force=True)
#run full MCMC sampler
stan_fit = stan_model.sample(data=fname_stan_data, chains=n_chains, 
                             iter_warmup=n_iter_warmup, iter_sampling=n_iter_sampling,
                             seed=1, refresh=10, max_treedepth=max_treedepth, adapt_delta=adapt_delta,
                             show_progress=True,  output_dir=dir_out+'stan_fit/')

#delete json files
fname_dir = np.array( os.listdir(dir_out) )
#velocity filenames
fname_json = fname_dir[ [bool(re.search('\.json$',f_d)) for f_d in fname_dir] ]
for f_j in fname_json: os.remove(dir_out + f_j)


#%% Postprocessing
### ======================================
#initiaize flatfile for sumamry of non-erg coefficinets and residuals
df_velinfo  = df_velprofs[['DSID', 'DSName','VelID','VelName','Vs30','Lat','Lon','Depth_MPt','Thk', 'Vs', 'flag_Z1']]
df_profinfo = df_velprofs[['DSID', 'DSName','VelID','VelName','Vs30','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)

# process regression output
# ---   ---   ---   ---
# Extract posterior samples
# - - - - - - - - - - - 
#hyper-parameters
col_names_hyp = ['s1','s2','s3','sigma_vel','tau_r']
#vel profile parameters
col_names_vs0 = ['Vs0.%i'%(k) for k in range(n_vel)]
col_names_k   = ['k.%i'%(k)   for k in range(n_vel)]
col_names_n   = ['n.%i'%(k)   for k in range(n_vel)]
col_names_dB  = ['r_dB.%i'%(k)  for k in range(n_vel)]
col_names_all = col_names_hyp + col_names_vs0 + col_names_k + col_names_n + col_names_dB
    
#extract raw hyper-parameter posterior samples
stan_posterior = np.stack([stan_fit.stan_variable(c_n) for c_n in col_names_hyp], axis=1)
#vel profile parameters
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('Vs0_p')),  axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('k_p')), axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('n_p')), axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('r_dB')),     axis=1)
    
#save raw-posterior distribution
df_stan_posterior_raw = pd.DataFrame(stan_posterior, columns = col_names_all)
df_stan_posterior_raw.to_csv(dir_out + fname_out_main + '_stan_posterior_raw' + '.csv', index=False)

# Summarize hyper-parameters
# - - - - - - - - - - - 
#summarize posterior distributions of hyper-parameters
#df_stan_posterior_raw = pd.read_csv('/home/hsalmanitehrani/GlobalRegression/all2/all2_stan_posterior_raw.csv')
perc_array = np.array([0.05,0.25,0.5,0.75,0.95])
df_stan_hyp = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_hyp = df_stan_hyp.append(df_stan_posterior_raw[col_names_hyp].mean(axis = 0), ignore_index=True)
df_stan_hyp.index = ['prc_%.2f'%(prc) for prc in perc_array]+['mean'] 
df_stan_hyp.to_csv(dir_out + fname_out_main + '_stan_hyperparameters' + '.csv', index=True)

#detailed posterior percentiles of posterior distributions
perc_array = np.arange(0.01,0.99,0.01)    
df_stan_posterior = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_posterior.index.name = 'prc'
df_stan_posterior.to_csv(dir_out + fname_out_main + '_stan_hyperposterior' + '.csv', index=True)

#del stan_posterior, col_names_all

# Velocity profile parameters
# - - - - - - - - - - - 
#shape parameters
# Vs0
param_vs0_mu  = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].mean()   for k in range(n_vel)])
param_vs0_med = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].median() for k in range(n_vel)])
param_vs0_sig = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].std()    for k in range(n_vel)])
# k
param_k_mu  = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].mean()   for k in range(n_vel)])
param_k_med = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].median() for k in range(n_vel)])
param_k_sig = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].std()    for k in range(n_vel)])
# n
param_n_mu  = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].mean()   for k in range(n_vel)])
param_n_med = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].median() for k in range(n_vel)])
param_n_sig = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].std()    for k in range(n_vel)])
# dB_r
###param_r_dB_mu  = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].mean()   for k in range(n_vel)])
###param_r_dB_med = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].median() for k in range(n_vel)])
###param_r_dB_sig = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].std()    for k in range(n_vel)])

#aleatory variability
param_sigma_vel_mu  = np.array([df_stan_posterior_raw.sigma_vel.mean()]   * n_vel)
param_sigma_vel_med = np.array([df_stan_posterior_raw.sigma_vel.median()] * n_vel)
param_sigma_vel_sig = np.array([df_stan_posterior_raw.sigma_vel.std()]    * n_vel)

#between event term r scaling
param_dBr_mu  = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].mean()   for k in range(n_vel)])
param_dBr_med = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].median() for k in range(n_vel)])
param_dBr_sig = np.array([df_stan_posterior_raw.loc[:,f'r_dB.{k}'].std()    for k in range(n_vel)])

#summarize parameters
params_summary = np.vstack((param_vs0_mu,
                            param_k_mu, 
                            param_n_mu,
                            param_dBr_mu,
                            param_vs0_med,
                            param_k_med, 
                            param_n_med,
                            param_dBr_med,
                            param_vs0_sig,
                            param_k_sig, 
                            param_n_sig,
                            param_dBr_sig,
                            param_sigma_vel_mu,
                            param_sigma_vel_med,
                            param_sigma_vel_sig,)).T
columns_names = ['param_vs0_mean', 'param_k_mean',  'param_n_mean', 'param_dBr_mean',
                 'param_vs0_med',  'param_k_med',   'param_n_med',  'param_dBr_med', 
                 'param_vs0_std',  'param_k_std',   'param_n_std',  'param_dBr_std',
                 'sigma_vel_mean', 'sigma_vel_med', 'sigma_vel_std']
df_params_summary = pd.DataFrame(params_summary, columns = columns_names, index=df_profinfo.index)
#create dataframe with parameters summary
df_params_summary = pd.merge(df_profinfo, df_params_summary, how='right', left_index=True, right_index=True)
df_params_summary[['DSID','VelID']] = df_params_summary[['DSID','VelID']].astype(int)
df_params_summary.to_csv(dir_out + fname_out_main + '_stan_parameters' + '.csv', index=False)

# Velocity profile prediction
# - - - - - - - - - - -
#mean scaling terms
#k scaling
#r1_new = df_stan_hyp.loc['prc_0.50','r1']
#n scaling
s1_new = df_stan_hyp.loc['prc_0.50','s1']
s2_new = df_stan_hyp.loc['prc_0.50','s2']
s3_new = df_stan_hyp.loc['prc_0.50','s3']
#between event term
dB_r_new = param_dBr_med
#mean profile parameters
param_k_new   = np.array([1]*len(vel_inv))#np.exp( dB_r_new[vel_inv] )
param_n_new   = (s1_new/(1+s2_new*((0.001*df_velprofs.Vs30.values)**(-s3_new))))+1;
param_a_new   =-1/param_n_new
param_vs0_new = (param_k_new*(param_a_new+1)*z_star + (1+param_k_new*(30-z_star))**(param_a_new+1) - 1) / (30*(param_a_new+1)*param_k_new) * df_velprofs.Vs30.values
#considering between event effects
param_kdBr_new   =  np.exp(dB_r_new[vel_inv]) 
param_vs0dBr_new = (param_kdBr_new*(param_a_new+1)*z_star + (1+param_kdBr_new*(30-z_star))**(param_a_new+1) - 1) / (30*(param_a_new+1)*param_kdBr_new) * df_velprofs.Vs30.values

#orignal profile parameters
param_k_orig   = np.exp(r1_orig*((df_velprofs.Vs30.values)**r2_orig) + r3_orig)
param_n_orig   = s1_orig*np.exp(s2_orig*df_velprofs.Vs30.values) + s3_orig*np.exp(s4_orig*df_velprofs.Vs30.values)
param_vs0_orig = p1_orig*(df_velprofs.Vs30.values)**2 + p2_orig*df_velprofs.Vs30.values + p3_orig

#mean prediction
y_data  = stan_data['Y']
y_new   = np.log(param_vs0_new    * ( 1 + param_k_new    * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_new))
y_newdB = np.log(param_vs0dBr_new * ( 1 + param_kdBr_new * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_new))
y_orig  = np.log(param_vs0_orig   * ( 1 + param_k_orig   * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_orig))
    
#compute residuals
res_tot     = y_data - y_new
res_orig    = y_data - y_orig
#within-event residuals
res_dW      = y_data - y_newdB
#between event residuals
res_dB      = res_tot - res_dW

#summary predictions and residuals
predict_summary = np.vstack((np.exp(y_new), res_tot,res_dW, res_dB, res_orig,)).T
columns_names   = ['VsProf_mean','res_tot','res_dW','res_dB','res_orig']
df_predict_summary = pd.DataFrame(predict_summary, columns = columns_names, index=df_velprofs.index)
#create dataframe with predictions and residuals
df_predict_summary = pd.merge(df_velinfo, df_predict_summary, how='right', left_index=True, right_index=True)
df_predict_summary[['DSID','VelID']] = df_predict_summary[['DSID','VelID']].astype(int)
df_predict_summary.to_csv(dir_out + fname_out_main + '_stan_residuals' + '.csv', index=False)


#%% Comparison
### ======================================

# Total Residual
# ---------------------------
#total residuals versus depth
i_sort    = np.argsort( df_predict_summary.Depth_MPt.values )
x_data    = df_predict_summary.Depth_MPt[i_sort]
y_data    = df_predict_summary.res_tot[i_sort]
x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
spl_mmean = interp.UnivariateSpline(x_data,y_data)
#spl_mmean.set_smoothing_factor(2)
y_mmean   = spl_mmean(x_mmean)

fname_fig = (fname_out_main + '_total_residuals_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(y_data, x_data, 'o',  markersize=4)
hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
#edit properties
ax.set_xlabel('total residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Total Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#total residuals versus depth (orignal model)
fname_fig = (fname_out_main + '_total_residuals_versus_depth_original').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_predict_summary.loc[:,'res_orig'], df_predict_summary.loc[:,'Depth_MPt'], 'o', label='Shi and Asimaki, 2018', zorder=1)
#edit properties
ax.set_xlabel('total residuals',   fontsize=30)
ax.set_ylabel('Depth (m)',   fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
#ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Total Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


#total residuals versus depth (proposed and orignal model)
fname_fig = (fname_out_main + '_total_residuals_versus_depth_comparison').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_predict_summary.loc[:,'res_tot'], df_predict_summary.loc[:,'Depth_MPt'],  'o', markersize=4, label='Proposed Model')
hl = ax.plot(df_predict_summary.loc[:,'res_orig'], df_predict_summary.loc[:,'Depth_MPt'], 'o', label='Shi and Asimaki, 2018', zorder=1)
#edit properties
ax.set_xlabel('total residuals',   fontsize=30)
ax.set_ylabel('Depth (m)',   fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Total Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#total residuals versus Vs30
i_sort    = np.argsort( df_predict_summary.Vs30.values )
x_data    = df_predict_summary.Vs30[i_sort]
y_data    = df_predict_summary.res_tot[i_sort]
x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
spl_mmean = interp.UnivariateSpline(x_data,y_data)
#spl_mmean.set_smoothing_factor(2)
y_mmean   = spl_mmean(x_mmean)

fname_fig = (fname_out_main + '_total_residuals_versus_Vs30').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(y_data,  x_data,  'o',  markersize=4)
hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
#edit properties
ax.set_xlabel('total residuals',  fontsize=30)
ax.set_ylabel(r'$V_{S30}$', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 2000])
ax.set_title(r'Total Residuals versus $V_{S30}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#total residuals versus depth (Vs30 bins)
vs30_bins = [(0,100),(100,250),(250, 400), (400, 800), (800, 3000)]
for k, vs30_b in enumerate(vs30_bins):
    i_binned = np.logical_and(df_predict_summary.Vs30 >= vs30_b[0],
                                             df_predict_summary.Vs30 <  vs30_b[1])
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_tot[i_sort]
    x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
    spl_mmean = interp.UnivariateSpline(x_data,y_data)
    #spl_mmean.set_smoothing_factor(2)
    y_mmean   = spl_mmean(x_mmean)

    fname_fig = (fname_out_main + '_total_residuals_versus_depth_vs30_%i_%i'%vs30_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl = ax.plot(y_data, x_data, 'o',  markersize=4)
    hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
    #edit properties
    ax.set_xlabel('total residuals',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    ax.set_title('Total Residuals versus Depth \n $V_{S30}= [%.i, %.i)$ (m/sec)'%vs30_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )




# Within profile Residual
# ---------------------------
#within-profile residuals versus depth
i_sort    = np.argsort( df_predict_summary.Depth_MPt.values )
x_data    = df_predict_summary.Depth_MPt[i_sort]
y_data    = df_predict_summary.res_dW[i_sort]
x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
spl_mmean = interp.UnivariateSpline(x_data,y_data)
#spl_mmean.set_smoothing_factor(2)
y_mmean   = spl_mmean(x_mmean)

fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(y_data, x_data, 'o',  markersize=4)
hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
#edit properties
ax.set_xlabel('within prof. residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Within-profile Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


#within-profile residuals versus depth (proposed and orignal model)
fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth_comparison').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_predict_summary.loc[:,'res_dW'], df_predict_summary.loc[:,'Depth_MPt'],  'o', markersize=4, label='Proposed Model')
hl = ax.plot(df_predict_summary.loc[:,'res_orig'], df_predict_summary.loc[:,'Depth_MPt'], 'o', label='Shi and Asimaki, 2018', zorder=1)
#edit properties
ax.set_xlabel('within prof. residuals',   fontsize=30)
ax.set_ylabel('Depth (m)',   fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Within-profile Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus Vs30
i_sort    = np.argsort( df_predict_summary.Vs30.values )
x_data    = df_predict_summary.Vs30[i_sort]
y_data    = df_predict_summary.res_dW[i_sort]
x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
spl_mmean = interp.UnivariateSpline(x_data,y_data)
#spl_mmean.set_smoothing_factor(2)
y_mmean   = spl_mmean(x_mmean)

fname_fig = (fname_out_main + '_withinprof_residuals_versus_Vs30').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(y_data,  x_data,  'o',  markersize=4)
hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
#edit properties
ax.set_xlabel('within prof. residuals',  fontsize=30)
ax.set_ylabel(r'$V_{S30}$', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 2000])
ax.set_title(r'Within-profile Residuals versus $V_{S30}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus depth (Vs30 bins)
vs30_bins = [(0,100),(100,250),(250, 400), (400, 800), (800, 3000)]
for k, vs30_b in enumerate(vs30_bins):
    i_binned = np.logical_and(df_predict_summary.Vs30 >= vs30_b[0],
                                             df_predict_summary.Vs30 <  vs30_b[1])
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_dW[i_sort]
    x_mmean   = np.linspace(x_data.min(), x_data.max(), 1000)
    spl_mmean = interp.UnivariateSpline(x_data,y_data)
    #spl_mmean.set_smoothing_factor(2)
    y_mmean   = spl_mmean(x_mmean)

    fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth_vs30_%i_%i'%vs30_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl = ax.plot(y_data, x_data, 'o',  markersize=4)
    hl = ax.plot(y_mmean, x_mmean, '-',  linewidth=2)
    #edit properties
    ax.set_xlabel('within prof. residuals',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    ax.set_title('Within-profile Residuals versus Depth \n $V_{S30}= [%.i, %.i)$ (m/sec)'%vs30_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )


# Parameter Scaling
# ---------------------------
vs30_array = np.logspace(np.log10(10), np.log10(2000))
#scaling relationships
#param_k_scl   = np.full(vs30_array.shape, np.exp(r1_new))
param_n_scl   = (s1_new/(1+s2_new*((0.001*vs30_array)**(-s3_new))))+1;
param_a_scl   =-1/param_n_scl
#param_vs0_scl1 = ((param_a_scl+1)*z_star + (1+1*(30-z_star))**(param_a_scl+1) - 1) / (30*(param_a_scl+1)) * vs30_array 
param_vs0_scl = (1.*(param_a_scl+1)*z_star + (1+1*(30-z_star))**(param_a_scl+1) - 1) / (30*(param_a_scl+1)*1.) * vs30_array 

# =============================================================================
# #scaling k
# fname_fig = (fname_out_main + '_scaling_param_k').replace(' ','_')
# fig, ax = plt.subplots(figsize = (10,10))
# hl = ax.semilogx(vs30_array, param_k_scl, '-', linewidth=4, zorder=10, label='Global Model')
# hl = ax.semilogx(df_params_summary.Vs30, df_params_summary.param_k_med, 'o',  linewidth=2, color='orangered', label='Posterior Median')
# hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_k_med, yerr=df_params_summary.param_k_std, 
#                  capsize=4, fmt='none', ecolor='orangered', label='Posterior Std')
# #edit properties
# ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
# ax.set_ylabel(r'$k$',       fontsize=30)
# ax.legend(loc='upper left', fontsize=30)
# ax.grid(which='both')
# ax.tick_params(axis='x', labelsize=25)
# ax.tick_params(axis='y', labelsize=25)
# ax.set_ylim([0, 10])
# ax.set_title(r'$k$ Scaling', fontsize=30)
# fig.tight_layout()
# fig.savefig( dir_fig + fname_fig + '.png' )
# =============================================================================

# =============================================================================
# #scaling k LOG
# fname_fig = (fname_out_main + '_scaling_param_k_log').replace(' ','_')
# fig, ax = plt.subplots(figsize = (10,10))
# hl = ax.semilogx(vs30_array, param_k_scl, '-', linewidth=4, zorder=10, label='Global Model')
# hl = ax.semilogx(df_params_summary.Vs30, df_params_summary.param_k_med, 'o',  linewidth=2, color='orangered', label='Posterior Median')
# hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_k_med, yerr=df_params_summary.param_k_std, 
#                  capsize=4, fmt='none', ecolor='orangered', label='Posterior Std')
# #edit properties
# ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
# ax.set_ylabel(r'$k$',       fontsize=30)
# ax.legend(loc='upper left', fontsize=30)
# ax.grid(which='both')
# ax.tick_params(axis='x', labelsize=25)
# ax.tick_params(axis='y', labelsize=25)
# #ax.set_ylim([0, 10])
# ax.set_title(r'$k$ Scaling', fontsize=30)
# fig.tight_layout()
# plt.yscale("log")
# fig.savefig( dir_fig + fname_fig + '.png' )
# =============================================================================

#scaling n
fname_fig = (fname_out_main + '_scaling_param_n').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.semilogx(vs30_array, param_n_scl, '-', linewidth=4, zorder=10, label='Global Model')
hl = ax.semilogx(df_params_summary.Vs30, df_params_summary.param_n_med, 'o',  linewidth=2, color='orangered', label='Posterior Median')
hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_n_med, yerr=df_params_summary.param_n_std, 
                  capsize=4, fmt='none', ecolor='orangered', label='Posterior Std')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$n$',       fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_title(r'$n$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#scaling Vs0
fname_fig = (fname_out_main + '_scaling_param_vs0').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.loglog(vs30_array, param_vs0_scl, '-', linewidth=4, zorder=10, label='Global Model')
hl = ax.loglog(df_params_summary.Vs30, df_params_summary.param_vs0_med, 'o',  linewidth=2, color='orangered', label='Posterior Median')
hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_vs0_med, yerr=df_params_summary.param_vs0_std, 
                  capsize=4, fmt='none', ecolor='orangered', label='Posterior Std')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$V_{S0}$ (m/sec)',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_title(r'$V_{S0}$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#delta B r
fname_fig = (fname_out_main + '_scatter_param_dBr').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_params_summary.Vs30, df_params_summary.param_dBr_med, 'o',  linewidth=2, color='orangered', label='Posterior Median')
hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_dBr_med, yerr=df_params_summary.param_dBr_std, 
                  capsize=4, fmt='none', ecolor='orangered', label='Posterior Std')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_ylim([-10, 10])
ax.set_title(r'$\delta B_r$ Scatter ', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Spatial variability
# ---------------------------
#pl_win = [[36, 39], [-124, -121]]
pl_win = [[37, 38.2], [-122.6, -121.4]]
#median delta B r
data2plot = df_params_summary[['Lat','Lon','param_dBr_med']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_med').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-6,  cmax=6, flag_grid=True, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', marker_size=30.)
#edit figure properties
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'Median $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
ax.set_title(r'$\delta B_r$ Spatial Variability ', fontsize=30)
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#std delta B r
data2plot = df_params_summary[['Lat','Lon','param_dBr_std']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_std').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=0,  cmax=2, flag_grid=True, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='Reds', marker_size=30.)


#edit figure properties
gl.xlabel_style = {'size': 25}
gl.ylabel_style = {'size': 25}
cbar.ax.tick_params(labelsize=15)
cbar.set_label(r'Std $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
ax.set_title(r'$\delta B_r$ Spatial Variability ', fontsize=30)
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Iterate Profiles
# ---------------------------
#iterate over vel profiles
for k, v_id_dsid in enumerate(vel_id_dsid):
    print('Plotting vel. profile %i of %i'%(k+1, n_vel))
    #extract vel profile of interest
    df_vel = df_predict_summary.loc[np.logical_and(df_velprofs.DSID==v_id_dsid[0], df_velprofs.VelID==v_id_dsid[1]), ]
    #profile name
    name_vel =  df_vel.DSName.values[0]+' '+df_vel.VelName.values[0]
    fname_vel = (fname_out_main + '_vel_prof_' + name_vel).replace(' ','_')

    #velocity prof information
    vs30 = df_vel.Vs30.values[0]
    #param_k      = np.exp( d )
    param_n      = (s1_new/(1+s2_new*((0.001*vs30)**(-s3_new))))+1;
    param_a      =-1/param_n
    #param_vs0    = (param_k*(param_a+1)*z_star    + (1+param_k*(30-z_star))**(param_a+1) - 1)    / (30*(param_a+1)*param_k) * vs30
    param_kdBr = np.exp(dB_r_new[k])
    param_vs0dBr = (param_kdBr*(param_a+1)*z_star + (1+param_kdBr*(30-z_star))
                    ** (param_a+1) - 1) / (30*(param_a+1)*param_kdBr) * vs30

    #velocity profile
    depth_array = np.concatenate([[0], np.cumsum(df_vel.Thk)])
    vel_array   = np.concatenate([df_vel.Vs.values, [df_vel.Vs.values[-1]]])
    #velocity model
    depth_m_array = np.linspace(depth_array.min(), depth_array.max(), 1000)
    #vel_m_array   = param_vs0    * ( 1 + param_k    * ( np.maximum(0, depth_m_array-z_star) ) )**(1/param_n)
    vel_mdB_array = param_vs0dBr * ( 1 + param_kdBr * ( np.maximum(0, depth_m_array-z_star) ) )**(1/param_n)
    
    #create figure   
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.step(vel_array,     depth_array, label='Velocity Profile')
    #hl2 = ax.plot(vel_m_array,   depth_m_array, linewidth=2.0, linestyle='dashed', label=r'Velocity model (w/o $\delta B_r$)')
    hl3 = ax.plot(vel_mdB_array, depth_m_array, linewidth=2.0,                     label=r'Velocity model (w/ $\delta B_r$)')
    #edit properties
    ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.legend(loc='lower left', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    #ax.set_xlim([0, 1500])
    #ax.set_ylim([0, 200])
    ax.invert_yaxis()
    ax.set_title(name_vel, fontsize=30)
    fig.tight_layout()
    fig.savefig( dir_fig + fname_vel + '.png' )
    

# Summary regression
# ---------------------------
#save summary statistics
stan_summary_fname = dir_out  + fname_out_main + '_stan_summary' + '.txt'
with open(stan_summary_fname, 'w') as f:
    print(stan_fit, file=f)

#create and save trace plots
dir_fig = dir_out  + 'summary_figs/'
#create figures directory if doesn't exit
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#create stan trace plots
stan_az_fit = az.from_cmdstanpy(stan_fit)
for c_name in col_names_hyp:
    #create trace plot with arviz
    ax = az.plot_trace(stan_az_fit,  var_names=c_name, figsize=(10,5) ).ravel()
    ax[0].yaxis.set_major_locator(plt_autotick())
    ax[0].set_xlabel('sample value')
    ax[0].set_ylabel('frequency')
    ax[0].set_title('')
    ax[0].grid(axis='both')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('sample value')
    ax[1].grid(axis='both')
    ax[1].set_title('')
    fig = ax[0].figure
    fig.suptitle(c_name)
    fig.savefig(dir_fig + fname_out_main + '_stan_traceplot_' + c_name + '_arviz' + '.png')

#summary residuals
#print('Total Residuals Mean: %.2f'%df_predict_summary.loc[:,'res_tot'].mean())
#print('Total Residuals Std Dev: %.2f'%df_predict_summary.loc[:,'res_tot'].std())
print('Within-profile Residuals Mean: %.2f'%df_predict_summary.loc[:,'res_dW'].mean())
print('Within-profile Residuals Std Dev: %.2f'%df_predict_summary.loc[:,'res_dW'].std())
