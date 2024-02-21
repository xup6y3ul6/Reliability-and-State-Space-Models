data {
  int<lower=1> T;
  vector[T] y;
  real m_0;
  cov_matrix[1] C_0;
}

parameters {
  real mu;
  real theta_0;
  vector[T] theta;
  real Phi;
  cov_matrix[1] R;
  cov_matrix[1] Q;
}

model {
  theta_0 ~ normal(m_0, sqrt(C_0[1, 1]));
  
  theta[1] ~ normal(Phi * theta_0, sqrt(Q[1, 1]));
  y[1] ~ normal(mu + theta[1], sqrt(R[1, 1]));
  
  for (t in 2:T) {
    theta[t] ~ normal(Phi * theta[t-1], sqrt(Q[1, 1]));
    y[t] ~ normal(mu + theta[t], sqrt(R[1, 1]));
  }
}

generated quantities {
  vector[T] y_hat;
  
  y_hat = mu + theta;
}
