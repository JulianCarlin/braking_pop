data { 
  int<lower=1> N;         // total number of samples
  vector[N] n_meas;       // all measured braking indices
  vector[N] zeta_meas;    // all measured zeta
  vector[N] zeta_unc;     // uncertainty on zeta
}
parameters {
  real<lower=2, upper=7> n_mu; // upper and lower bounds set implicit uniform prior
  vector<lower=0>[N] zeta;      // treat zeta as missing data with one measurement (zeta_meas) and uncertainty
  real<lower=0> n_sig;          // population spread, can set to zero if want delta-function population distribution
}
model {
  n_sig ~ cauchy(0, 5);         // broad uninformative prior on n_sig
  zeta ~ normal(zeta_meas, zeta_unc);  // this incorporates zeta measurement error
  n_meas ~ normal(n_mu, sqrt(n_sig^2 + zeta^2)); // this is the model, the n_sig and zeta are packaged like this for sampling efficiency
}
