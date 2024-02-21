#include ssm-function.stan

data {
  int<lower=1> N; // number of subjects
  array[N] int<lower=1> T; // number of observation for each subject
  int<lower=1> max_T; // maximum number of observation
  int<lower=1> P; // number of affects
  array[N] matrix[P, max_T] y; // observations 
  array[N] vector[P] m_0; // prior mean of the intial state
  array[N] cov_matrix[P] C_0; // prior covariance of the intial state
}

parameters {
  array[N] vector[P] mu; // ground mean/trane
  array[N] vector[P] theta_0; // initial latent state
  array[N] matrix[P, max_T] theta; // latent states
  array[N] matrix[P, P] Phi; // autoregressive parameters
  
  array[N] cholesky_factor_corr[P] L_Omega_R; 
  array[N] cholesky_factor_corr[P] L_Omega_Q;
  array[N] vector<lower=0>[P] tau_R;
  array[N] vector<lower=0>[P] tau_Q;
  
  vector[P] gamma_mu; // prior mean of the ground mean
  cov_matrix[P] Psi_mu; // prior covariance of the ground mean
  vector[P * P] gamma_Phi; // prior mean of the autoregressive parameters
  cov_matrix[P * P] Psi_Phi; // prior covariance of the autoregressive parameters
  
  vector<lower=0>[P] loc_tau_R; // location of the measurement error
  vector<lower=0>[P] loc_tau_Q; // location of the innovation noise
  vector<lower=0>[P] scale_tau_R; // scale of the measurement error
  vector<lower=0>[P] scale_tau_Q; // scale of the innovation noise
  real<lower=0> eta_Omega_R; // shape of the LKJ prior for the measurement error
  real<lower=0> eta_Omega_Q; // shape of the LKJ prior for the innovation noise
}

transformed parameters{
  array[N] cov_matrix[P] R; // covariance of the measurment error
  array[N] cov_matrix[P] Q; // covariance of the innovation noise
  for (n in 1:N) {
    R[n] = diag_pre_multiply(tau_R[n], L_Omega_R[n]) * diag_pre_multiply(tau_R[n], L_Omega_R[n])';
    Q[n] = diag_pre_multiply(tau_Q[n], L_Omega_Q[n]) * diag_pre_multiply(tau_Q[n], L_Omega_Q[n])';
  }
}

model {
  // level 1 (within subject)
  for (n in 1:N) {
    // when t = 0
    theta_0[n] ~ multi_normal(m_0[n], C_0[n]);
  
    // when t = 1
    theta[n][, 1] ~ multi_normal(Phi[n] * theta_0[n], Q[n]);
    y[n][, 1] ~ multi_normal(mu[n] + theta[n][, 1], R[n]);
    
    // when t = 2, ..., T 
    for (t in 2:T[n]) {
      theta[n][, t] ~ multi_normal(Phi[n] * theta[n][, t - 1], Q[n]);
      y[n][, t] ~ multi_normal(mu[n] + theta[n][, t], R[n]);
    }
  }
  
  
  
  // level 2 (between subject)
  for (n in 1:N) {
    mu[n] ~ multi_normal(gamma_mu, Psi_mu);
    to_vector(Phi[n]) ~ multi_normal(gamma_Phi, Psi_Phi);
    
    
    tau_R[n] ~ cauchy(loc_tau_R, scale_tau_R);
    tau_Q[n] ~ cauchy(loc_tau_Q, scale_tau_Q);
    L_Omega_R[n] ~ lkj_corr_cholesky(eta_Omega_R);
    L_Omega_Q[n] ~ lkj_corr_cholesky(eta_Omega_Q);
    
  }
  
  // the (hyper)priors of parameters are set as the Stan default values
}

generated quantities {
  array[N] matrix[P, max_T] y_hat;
  array[N] matrix[P, P] Tau; 
  array[N] vector[P] rel_W;
  vector[P] rel_B;
  
  for (n in 1:N) {
    // prediction 
    for (t in 1:T[n]) {
      y_hat[n][, t] = mu[n] + theta[n][, t];
    }
    
    // within-subject reliability
    Tau[n] = to_matrix((identity_matrix(P * P) - kronecker_prod(Phi[n], Phi[n])) \ to_vector(Q[n]), P, P);
  
    for (p in 1:P) {
      rel_W[n, p] = Tau[n, p, p] / (Tau[n, p, p] + R[n, p, p]);
    }
  }
  
  // between-subject reliability
  for (p in 1:P) {
    rel_B[p] = Psi_mu[p, p] / (Psi_mu[p, p] + mean(Tau[, p, p]) + loc_tau_R[p]^2);
  }
}
