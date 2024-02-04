data {

  int N; //number of points
  int NVEL; //number of velocity profiles
  
  //z^star parameter
  real z_star;
  
  //velocity array indices 
  int<lower=1,upper=NVEL> i_vel[N];
  
  //input arrays
  vector[NVEL] Vs30;  
  //observations
  vector[N] Z; //Depth array 
  vector[N] Y; //Velocity array

}

transformed data {
  real delta = 1e-9;
}

parameters {
  //model parameters
  //k scaling
  real <lower=0>r1;
  real <lower=0>r2;
  real <lower=0>r3;
  real <lower=0>r4;
  //s scaling
  real<lower=0.0> s1;
  real<lower=0.0> s2;
  real<lower=0.0> s3;
 
  //aleatory variability
  real<lower=0.0> sigma_vel;
  //between event std
  real<lower=0.0> tau_r;

  //between event term
  vector[NVEL] r_dB;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;

  //model parameters
  k_p   = exp((r1/(1+r2*((0.001*Vs30)^-r3)))-r4+r_dB);
  //n_p   = 1. + s3 * inv_logit( (log(Vs30)-s1) * s2 );
  n_p   = (s1/(1+s2*((0.001*Vs30)^-s3)))+1;
  a_p   =-1. ./ n_p;
  for(i in 1:NVEL){
      // Vs0 = (k_p*(a_p+1.)*z_star + (1.+k_p*(30.-z_star))^(a_p+1.) - 1.) / (30.*(a_p+1.)*k_p) * Vs30;
      Vs0_p[i] = (k_p[i]*(a_p[i]+1.)*z_star + (1.+k_p[i]*(30.-z_star))^(a_p[i]+1.) - 1.) / (30.*(a_p[i]+1.)*k_p[i]) * Vs30[i];
  }

}

model {
  //velocity model
  vector[N] VEL_M;

  //prior distributions
  //---   ---   ---   ---   ---   ---
  //model coefficients
  r1 ~ lognormal(1.4,0.25);
  r2 ~ lognormal(-8,0.25);
  r3 ~ lognormal(3.5,0.1);
  r4 ~ lognormal(0.08,0.05);
  s1 ~ lognormal(1.9,0.3);
  s2 ~ lognormal(-0.3,0.45);
  s3 ~ lognormal(0.8,0.15);
  
  //aleatory variability
  sigma_vel ~ lognormal(-0.3,0.6);
  tau_r     ~ lognormal(-0.3,1);
 
  //between event term r scaling
  r_dB ~ normal(0,tau_r);
  
 
  //functional model
  //---   ---   ---   ---   ---   ---
  //evaluate velocity model
  for(i in 1:N){
    int i_v = i_vel[i];
    VEL_M[i] =  log(Vs0_p[i_v]) + log( (1. + k_p[i_v] * max([0., Z[i]-z_star]) )^(1./n_p[i_v]) );
  }
   
  //likelihood function
  //---   ---   ---   ---   ---   ---
  //model likelihood
  Y ~ normal(VEL_M,sigma_vel);
}
