data {
    # Sizes
    int<lower=1> N;
    int<lower=1> K;
    # Data
    vector[N] y;
    vector[K] kern;
    vector[N+K-1] t_eval;
    # Priors
    #
    real mu_dy;
    real mu_y0;
    real<lower=0> mu_k;
    real mu_t0;
    # Standard deviations
    real<lower=0> sigma_dy;
    real<lower=0> sigma_y0;
    real<lower=0> sigma_t0;

}
parameters {
    real dy;
    real<lower=0> k;
    real y0;
    real<lower=0> sigma;
    real t0;
}
model {
    vector[N+K-1] y_eval;
    vector[N] y_conv;

    sigma ~ exponential(0.1);
    t0 ~ normal(mu_t0, sigma_t0);
    dy ~ normal(mu_dy, sigma_dy);
    k ~ exponential(mu_k);
    y0 ~ normal(mu_y0, sigma_y0);

    for (i in 1:(N+K-1)) {
        if (t_eval[i] >= t0) {
            y_eval[i] <- y0 - dy * expm1(-k * (t_eval[i] - t0));
        } else {
            y_eval[i] <- y0;
        } 
    }

    for (i in 1:N) {
        y_conv[i] <- dot_product(kern, segment(y_eval, i, K));
    }

    y ~ normal(y_conv, sigma);
}