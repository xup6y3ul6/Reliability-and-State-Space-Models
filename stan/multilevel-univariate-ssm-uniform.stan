data {
  int<lower=0> N;
  array[N] int<lower=1> T;
  int<lower=1> max_T;
  array[N] vector[max_T] y;
}

transformed data {
  real m_0 = 50.0;
  real c_0 = sqrt(1000);
}

parameters {
  // parameters
  vector[N] theta_0;
  array[N] vector[max_T] theta;
  vector[N] mu;
  vector<lower=-1, upper=1>[N] phi;
  vector<lower=0>[N] sigma_epsilon; // sd of the measurement error
  vector<lower=0>[N] sigma_omega;
  
  
  // hyperparameters
  real gamma_mu;
  real<lower=0> psi_mu;
  real gamma_phi;
  real<lower=0> psi_phi;
  //real<lower=0> gamma_sigma2_epsilon;
  //real<lower=0> gamma_sigma2_omega;
  
}

transformed parameters {
  
}


model {
  // level 1 (within subject)
  for (n in 1:N) {
    // when t = 0
    theta_0[n] ~ normal(m_0, c_0);
    // when t = 1
    theta[n][1] ~ normal(phi[n] * theta_0[n], sigma_omega[n]);
    // when t = 2, ..., T
    theta[n][2:T[n]] ~ normal(phi[n] * theta[n][1:(T[n]-1)], sigma_omega[n]);
    
    y[n][1:T[n]] ~ normal(mu[n] + theta[n][1:T[n]], sigma_epsilon[n]);
  }
  
  // level 2 (between subject)
  mu ~ normal(gamma_mu, psi_mu);
  phi ~ normal(gamma_phi, psi_phi);i
  sigma_epsilon ~ uniform(0, 100);
  sigma_omega ~ uniform(0, 100);
  
  
}
  

generated quantities {
  array[N] vector[max_T] y_hat;
  vector[N] tau2;
  vector[N] rel_W;
  real rel_B;
  
  for (n in 1:N) {
    y_hat[n] = mu[n] + theta[n];
  }
  
  tau2 = sigma_omega^2 ./ (1.0 - phi^2);
  rel_W = tau2 ./ (tau2 + sigma_epsilon^2);
  
  rel_B = psi_mu^2 / (psi_mu^2 + mean(tau2) + mean(sigma_epsilon^2));
}
