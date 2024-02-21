#include ssm-function.stan

data {
  int<lower=1> T;
  array[T] vector[2] y;
  vector[2] m_0;
  cov_matrix[2] C_0;
}

parameters {
  vector[2] mu;
  vector[2] theta_0;
  array[T] vector[2] theta;
  matrix[2, 2] Phi;
  cov_matrix[2] R;
  cov_matrix[2] Q;
}

model {
  theta_0 ~ multi_normal(m_0, C_0);
  
  theta[1] ~ multi_normal(Phi * theta_0, Q);
  y[1] ~ multi_normal(mu + theta[1], R);
  
  for (t in 2:T) {
    theta[t] ~ multi_normal(Phi * theta[t-1], Q);
    y[t] ~ multi_normal(mu + theta[t], R);
  }
  
}

generated quantities {
  array[T] vector[2] y_hat;
  matrix[2, 2] Tau; 
  real rel_1; 
  real rel_2;
  
  for (t in 1:T) {
    y_hat[t] = mu + theta[t];
  }
  
  Tau = to_matrix((identity_matrix(4) - kronecker_prod(Phi, Phi)) \ to_vector(Q), 2, 2);
  
  rel_1 = Tau[1, 1] / (Tau[1, 1] + R[1, 1]);
  rel_2 = Tau[2, 2] / (Tau[2, 2] + R[2, 2]);
}
