data { 
  int<lower=1> N;         // total number of samples
  vector[N] n_meas;       // all measured braking indices
  vector[N] zeta_meas;    // all measured zeta
  vector[N] zeta_unc;     // uncertainty on zeta
}
parameters {
  real<lower=0, upper=1> theta;    // mixture parameter
  real<lower=2, upper=7> n_mu1;     // upper and lower bounds set implicit uniform prior
  real<lower=n_mu1, upper=8> n_mu2; // upper and lower bounds set implicit uniform prior
  vector<lower=0>[2] n_sig;         // population spread, can set to zero if want delta-function population distribution
  vector<lower=0>[N] zeta;          // treat zeta as missing data with one measurement (zeta_meas) and uncertainty  
}
model {
  n_sig ~ cauchy(0, 5);         // broad uninformative prior on n_sig
  zeta ~ normal(zeta_meas, zeta_unc);  // this incorporates zeta measurement error
  for (n in 1:N) {
    real zetan = zeta[n]^2;
    target += log_mix(theta,
                      normal_lpdf(n_meas[n] | n_mu1, sqrt(n_sig[1]^2 + zetan)),
                      normal_lpdf(n_meas[n] | n_mu2, sqrt(n_sig[2]^2 + zetan))
    );
  }
}
