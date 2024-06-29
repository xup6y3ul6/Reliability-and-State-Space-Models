//include ssm-function.stan

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of observation for each subject
  int<lower=1> max_T; // maximum number of observation
  array[N] vector[max_T] y; // observations 
}

transformed data {
  real m_0; // prior mean of the intial state
  real<lower=0> c_0; // prior sd of the intial state
  
  // assign the mean and variance for the initial latent state's prior
  m_0 = 0.0;
  c_0 = sqrt(1000);
  
}

parameters {
  // parameters
  array[N] real mu; // ground mean/trane
  array[N] real theta_0; // initial latent state
  array[N] vector[max_T] theta; // latent states
  array[N] real<lower=0, upper=1> phi; // autoregressive parameters
  array[N] real<lower=0> sigma_epsilon; // sd of the measurment error
  array[N] real<lower=0> sigma_omega; // sd of the innovation noise
  
  // hyperparameters
  real gamma_mu; // prior mean of the ground mean
  real<lower=0> psi_mu; // prior sd of the ground mean
  real gamma_phi; // prior mean of the autoregressive parameters
  real<lower=0> psi_phi; // prior sd of the autoregressive parameters
  
  real gamma_sigma2_epsilon;
  real<lower=0> psi_sigma2_epsilon;
  real gamma_sigma2_omega;
  real<lower=0> psi_sigma2_omega;
}

transformed parameters {
  real mu_sigma2_epsilon;
  //real var_sigma2_epsilon;
  real mu_sigma2_omega;
  //real var_sigma2_omega;
  
  mu_sigma2_epsilon = exp(gamma_sigma2_epsilon + psi_sigma2_epsilon^2 / 2);
  //var_sigma2_epsilon = exp();
  mu_sigma2_omega = exp(gamma_sigma2_omega + psi_sigma2_omega^2 / 2);
  //var_sigma2_omega = exp;  
}

model {
  // level 1 (within subject)
  for (n in 1:N) {
    // when t = 0
    theta_0[n] ~ normal(m_0, c_0);
  
    // when t = 1
    theta[n][1] ~ normal(phi[n] * theta_0[n], sigma_omega[n]);
    y[n][1] ~ normal(mu[n] + theta[n][1], sigma_epsilon[n]);
    
    // when t = 2, ..., T 
    for (t in 2:T[n]) {
      theta[n][t] ~ normal(phi[n] * theta[n][t - 1], sigma_omega[n]);
      y[n][t] ~ normal(mu[n] + theta[n][t], sigma_epsilon[n]);
    }
  }
  
  // level 2 (between subject)
  mu ~ normal(gamma_mu, psi_mu);
  phi ~ normal(gamma_phi, psi_phi);
  sigma_epsilon^2 ~ lognormal(gamma_sigma2_epsilon, psi_sigma2_epsilon);
  sigma_omega^2 ~ lognormal(gamma_sigma2_omega, psi_sigma2_omega);
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  array[N] vector[max_T] y_hat;
  array[N] real tau2; 
  array[N] real rel_W;
  real rel_B;
  
  for (n in 1:N) {
    // prediction 
    for (t in 1:T[n]) {
      y_hat[n][t] = mu[n] + theta[n][t];
    }
    
    // within-subject reliability
    tau2[n] = sigma_omega[n]^2 / (1 - phi[n]^2);
    rel_W[n] = tau2[n] / (tau2[n] + sigma_epsilon[n]^2);
    
  }
  
  // between-subject reliability
  rel_B = psi_mu^2 / (psi_mu^2 + mean(tau2) + mu_sigma2_epsilon);
}
