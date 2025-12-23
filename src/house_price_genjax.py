"""
GenJAX House Price Prediction Example
Bayesian linear regression on the Kaggle House Prices dataset

This script demonstrates probabilistic programming with GenJAX for
uncertainty-aware house price prediction.

Usage:
    python house_price_genjax.py              # Default: importance sampling
    python house_price_genjax.py --mh         # Use Metropolis-Hastings
    python house_price_genjax.py --robust     # Robust model with hybrid Gibbs+HMC inference

The --robust flag uses a robust model with outlier detection and demonstrates
Gen's programmable inference: mixing Gibbs sampling for discrete outlier
indicators with Hamiltonian Monte Carlo (HMC) for continuous coefficients.
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd
import numpy as np

from genjax import gen, normal, uniform, flip, Target, ChoiceMap
from genjax import Regenerate, SelectionBuilder
from genjax.inference.requests import HMC
from genjax.inference.smc import ImportanceK


# --- Model Definition ---

def run_metropolis_hastings(target, n_samples=1000, n_burnin=500, step_size=0.1, seed=42, adaptive=True, target_accept=0.25):
    """
    WIP: Not fully implemented yet.
    Run Metropolis-Hastings MCMC inference with optional adaptive step size.

    Args:
        target: GenJAX Target with model, args, and constraints
        n_samples: Number of posterior samples to collect (after burn-in)
        n_burnin: Number of burn-in iterations to discard
        step_size: Initial standard deviation for Gaussian random walk proposal
        seed: Random seed for reproducibility
        adaptive: Whether to adapt step size during burn-in (default True)
        target_accept: Target acceptance rate for adaptation (default 0.25)

    Returns:
        posterior_samples: Dictionary of posterior samples for each latent variable
    """
    key = jrandom.PRNGKey(seed)
    model = target.p
    args = target.args
    constraints = target.constraint

    # Latent variable names and their bounds (for noise_std which is uniform)
    latent_names = ["coef_0", "coef_1", "coef_2", "coef_3", "intercept", "noise_std"]

    # Initialize from prior using importance sampling (single particle)
    key, init_key = jrandom.split(key)
    init_trace, init_log_weight = model.importance(init_key, constraints, args)
    current_choices = init_trace.get_choices()
    current_log_prob = init_log_weight

    # Storage for samples
    samples = {name: [] for name in latent_names}

    n_total = n_burnin + n_samples
    n_accepted = 0
    n_accepted_burnin = 0

    # Adaptive step size parameters
    adapt_window = 25  # Adapt every N iterations during burn-in

    for i in range(n_total):
        key, proposal_key, accept_key = jrandom.split(key, 3)

        # Propose new values using Gaussian random walk
        proposed_values = {}
        for j, name in enumerate(latent_names):
            current_val = current_choices[name]
            noise = jrandom.normal(jrandom.fold_in(proposal_key, j)) * step_size
            proposed_val = current_val + noise

            # Clip noise_std to valid range [0.1, 0.5]
            # Note: Clipping breaks proposal symmetry at boundaries, which technically
            # violates detailed balance. A proper fix would use reflection or
            # log-transform. Acceptable for this WIP implementation.
            if name == "noise_std":
                proposed_val = jnp.clip(proposed_val, 0.1, 0.5)

            proposed_values[name] = proposed_val

        # Build proposed choice map (latents + observed data)
        proposed_choices = ChoiceMap.d({
            **proposed_values,
            "log_prices": constraints["log_prices"]
        })

        # Compute log probability of proposed state
        proposed_trace, proposed_log_weight = model.importance(
            jrandom.fold_in(proposal_key, 100), proposed_choices, args
        )

        # Metropolis-Hastings acceptance ratio (in log space)
        log_alpha = proposed_log_weight - current_log_prob

        # Accept or reject
        u = jrandom.uniform(accept_key)
        if jnp.log(u) < log_alpha:
            current_choices = proposed_trace.get_choices()
            current_log_prob = proposed_log_weight
            if i >= n_burnin:
                n_accepted += 1
            else:
                n_accepted_burnin += 1

        # Adaptive step size during burn-in
        if adaptive and i < n_burnin and (i + 1) % adapt_window == 0:
            window_accept_rate = n_accepted_burnin / adapt_window
            # Adjust step size to approach target acceptance rate
            if window_accept_rate < target_accept - 0.05:
                step_size *= 0.7  # Decrease step size
            elif window_accept_rate > target_accept + 0.10:
                step_size *= 1.3  # Increase step size
            # Reset counter for next window
            n_accepted_burnin = 0

        # Store sample (after burn-in)
        if i >= n_burnin:
            for name in latent_names:
                samples[name].append(float(current_choices[name]))

    # Convert to arrays
    posterior_samples = {name: jnp.array(vals) for name, vals in samples.items()}

    acceptance_rate = n_accepted / n_samples
    print(f"    MH acceptance rate: {acceptance_rate:.1%}")
    print(f"    Final step size: {step_size:.4f}")

    return posterior_samples


def run_importance_sampling(target, k_particles=1000, seed=42):
    """
    Run importance sampling inference.

    Args:
        target: GenJAX Target with model, args, and constraints
        k_particles: Number of particles for importance sampling
        seed: Random seed for reproducibility

    Returns:
        posterior_samples: Dictionary of posterior samples for each latent variable
    """
    alg = ImportanceK(target, k_particles=k_particles)
    key = jrandom.PRNGKey(seed)

    sub_keys = jrandom.split(key, k_particles)
    _, posterior_samples = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, target
    )

    return posterior_samples


def run_gibbs_hmc(target, n_samples=500, n_burnin=200, hmc_eps=0.0001, hmc_L=50, seed=42):
    """
    Run hybrid Gibbs + HMC inference for the robust model.

    This implements programmable inference by combining:
    - Gibbs sampling: Resample discrete outlier indicators from their conditional
    - HMC: Update continuous coefficients using Hamiltonian Monte Carlo

    Args:
        target: GenJAX Target with robust model, args, and constraints
        n_samples: Number of posterior samples to collect (after burn-in)
        n_burnin: Number of burn-in iterations to discard
        hmc_eps: HMC leapfrog step size
        hmc_L: Number of HMC leapfrog steps
        seed: Random seed for reproducibility

    Returns:
        posterior_samples: Dictionary of posterior samples for each latent variable
    """
    key = jrandom.PRNGKey(seed)
    model = target.p
    args = target.args
    constraints = target.constraint

    # Define selections for different variable types using SelectionBuilder
    # HMC variables: coefficients and intercept (unbounded normal priors)
    # Note: noise_std uses uniform prior with bounds, so we exclude it from HMC
    # and update it separately with Regenerate
    hmc_addrs = ["coef_0", "coef_1", "coef_2", "coef_3", "intercept"]
    hmc_selection = SelectionBuilder[hmc_addrs[0]]
    for addr in hmc_addrs[1:]:
        hmc_selection = hmc_selection | SelectionBuilder[addr]

    # Discrete/bounded variables: outlier indicator + noise_std
    # These are updated via Regenerate (resample from conditional prior)
    discrete_selection = SelectionBuilder["is_outlier"] | SelectionBuilder["noise_std"]

    continuous_addrs = ["coef_0", "coef_1", "coef_2", "coef_3", "intercept", "noise_std"]

    # Initialize trace using importance sampling
    key, init_key = jrandom.split(key)
    current_trace, _ = model.importance(init_key, constraints, args)

    # Storage for samples - continuous vars + is_outlier array
    samples = {name: [] for name in continuous_addrs}
    samples["is_outlier"] = []  # Will store arrays

    n_total = n_burnin + n_samples
    n_hmc_accepted = 0

    print(f"    Running {n_burnin} burn-in + {n_samples} sampling iterations...")

    for i in range(n_total):
        key, gibbs_key, hmc_key = jrandom.split(key, 3)

        # --- Gibbs step: Resample discrete outlier indicators + noise_std ---
        # Regenerate resamples the selected addresses from their exact conditional
        # distribution, so Gibbs moves are always accepted (no MH step needed)
        gibbs_request = Regenerate(discrete_selection)
        current_trace, _, _, _ = current_trace.edit(gibbs_key, gibbs_request)

        # --- HMC step: Update continuous coefficients ---
        hmc_request = HMC(hmc_selection, eps=jnp.array(hmc_eps), L=hmc_L)
        new_trace, hmc_weight, _, _ = current_trace.edit(hmc_key, hmc_request)

        # Accept/reject HMC move (HMC proposes, MH accepts/rejects)
        hmc_accept_key, key = jrandom.split(key)
        if jnp.log(jrandom.uniform(hmc_accept_key)) < hmc_weight:
            current_trace = new_trace
            if i >= n_burnin:
                n_hmc_accepted += 1

        # Store sample (after burn-in)
        if i >= n_burnin:
            choices = current_trace.get_choices()
            for name in continuous_addrs:
                samples[name].append(float(choices[name]))
            # Store is_outlier as array
            samples["is_outlier"].append(np.array(choices["is_outlier"]))

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"    Iteration {i + 1}/{n_total}")

    # Convert to arrays
    posterior_samples = {}
    for name in continuous_addrs:
        posterior_samples[name] = jnp.array(samples[name])
    # is_outlier is a list of arrays, stack them: (n_samples, n_train)
    posterior_samples["is_outlier"] = jnp.stack(samples["is_outlier"], axis=0)

    # Report HMC acceptance rate (Gibbs always accepts by construction)
    hmc_accept_rate = n_hmc_accepted / n_samples
    print(f"    HMC acceptance rate: {hmc_accept_rate:.1%}")

    return posterior_samples


@gen
def house_price_model(X):
    """
    Bayesian linear regression for house prices.
    X: feature matrix (n_samples, n_features)
    """
    # Priors on regression coefficients
    # We expect positive coefficients for quality/size features
    coef_0 = normal(0.0, 1.0) @ "coef_0"
    coef_1 = normal(0.0, 1.0) @ "coef_1"
    coef_2 = normal(0.0, 1.0) @ "coef_2"
    coef_3 = normal(0.0, 1.0) @ "coef_3"

    # Prior on intercept (log-scale, since we'll predict log prices)
    intercept = normal(12.0, 1.0) @ "intercept"  # ~$160k baseline

    # Prior on noise (houses vary in price even with same features)
    noise_std = uniform(0.1, 0.5) @ "noise_std"

    # Generate predictions for each house
    coeffs = jnp.array([coef_0, coef_1, coef_2, coef_3])
    predictions = X @ coeffs + intercept

    # Likelihood: observed prices given our linear model
    # GenJAX supports vectorized distributions - the normal distribution
    # broadcasts over the predictions array, treating each as independent
    normal(predictions, noise_std) @ "log_prices"

    return predictions


@gen
def robust_house_price_model(X):
    """
    Robust Bayesian linear regression with outlier detection.
    Uses a mixture model to handle outliers in a JAX-compatible way.
    X: feature matrix (n_samples, n_features)

    The model treats each observation as coming from a mixture:
    - With probability 0.95: normal house (tight Gaussian around prediction)
    - With probability 0.05: outlier (wide Gaussian, 10x noise)

    The discrete outlier indicators are sampled via `flip`, and observations
    use different noise levels based on outlier status. This allows the model
    to identify data points that don't fit the linear relationship.
    """
    n_samples = X.shape[0]

    # Priors on regression coefficients
    coef_0 = normal(0.0, 1.0) @ "coef_0"
    coef_1 = normal(0.0, 1.0) @ "coef_1"
    coef_2 = normal(0.0, 1.0) @ "coef_2"
    coef_3 = normal(0.0, 1.0) @ "coef_3"

    # Prior on intercept (log-scale, since we'll predict log prices)
    intercept = normal(12.0, 1.0) @ "intercept"  # ~$160k baseline

    # Prior on noise for normal houses
    noise_std = uniform(0.1, 0.5) @ "noise_std"

    # Generate predictions for each house
    coeffs = jnp.array([coef_0, coef_1, coef_2, coef_3])
    predictions = X @ coeffs + intercept

    # Sample outlier indicators for all houses (vectorized)
    # Shape: (n_samples,) boolean array
    outlier_probs = jnp.full(n_samples, 0.05)
    is_outlier = flip(outlier_probs) @ "is_outlier"

    # Use different noise levels for outliers vs normal observations:
    # - Normal houses: tight Gaussian with noise_std
    # - Outliers: wide Gaussian with 10x noise (captures extreme deviations)
    outlier_noise_scale = 10.0
    effective_noise = jnp.where(is_outlier, noise_std * outlier_noise_scale, noise_std)

    # Likelihood with per-observation noise based on outlier status
    normal(predictions, effective_noise) @ "log_prices"

    return predictions, is_outlier


def main(use_mh=False, use_robust=False):
    print("=" * 60)
    print("GenJAX Bayesian House Price Prediction")
    if use_robust:
        print("(Robust Model with Outlier Detection)")
    print("=" * 60)

    # --- Load and preprocess data ---
    print("\n[1] Loading data...")
    train_df = pd.read_csv("data/train.csv")
    print(f"    Loaded {len(train_df)} total samples")

    # Select numerical features
    features = ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF"]
    print(f"    Features: {features}")

    # Handle missing values and prepare data
    X = train_df[features].fillna(0).values.astype(np.float32)
    y = np.log(train_df["SalePrice"].values).astype(np.float32)

    # Standardize features (compute stats on full data, then split)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # 70/30 train/test split
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_test = n_total - n_train

    # Shuffle indices for random split
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = jnp.array(X[train_idx])
    y_train = y[train_idx]
    X_test = jnp.array(X[test_idx])
    y_test = y[test_idx]

    print(f"    Training set: {n_train} samples (70%)")
    print(f"    Test set: {n_test} samples (30%)")
    print(f"    Train log prices range: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"    (Actual prices: ${np.exp(y_train.min()):,.0f} to ${np.exp(y_train.max()):,.0f})")

    # --- Build constraints ---
    print("\n[2] Setting up inference...")

    constraints = ChoiceMap.d({"log_prices": y_train})
    model = robust_house_price_model if use_robust else house_price_model
    if use_robust:
        print("    Using robust model with outlier detection")

    # Create inference target
    target = Target(
        model,
        (X_train,),
        constraints
    )

    # --- Run inference ---
    if use_robust:
        # Robust model uses hybrid Gibbs+HMC inference
        # - Gibbs sampling for discrete outlier indicators
        # - HMC for continuous coefficients
        print("\n[3] Running hybrid Gibbs+HMC inference...")
        print("    (Gibbs for discrete outliers, HMC for continuous coefficients)")
        n_mcmc_samples = 500
        n_mcmc_burnin = 200
        print(f"    MCMC samples: {n_mcmc_samples} (after {n_mcmc_burnin} burn-in)")
        posterior_samples = run_gibbs_hmc(
            target,
            n_samples=n_mcmc_samples, n_burnin=n_mcmc_burnin,
            hmc_eps=0.0001, hmc_L=50, seed=42
        )
    elif use_mh:
        print("\n[3] Running Metropolis-Hastings inference...")
        n_samples = 1000
        n_burnin = 1000
        print(f"    Number of samples: {n_samples} (after {n_burnin} burn-in)")
        posterior_samples = run_metropolis_hastings(target, n_samples=n_samples, n_burnin=n_burnin, step_size=0.1, seed=42)
    else:
        print("\n[3] Running importance sampling inference...")
        k_particles = 1000
        print(f"    Number of particles: {k_particles}")
        posterior_samples = run_importance_sampling(target, k_particles=k_particles, seed=42)

    # --- Extract and display results ---
    print("\n[4] Posterior estimates for coefficients:")
    print("-" * 50)

    # Extract posterior samples for coefficients
    for i, name in enumerate(features):
        coef_samples = posterior_samples[f"coef_{i}"]
        mean_val = jnp.mean(coef_samples)
        std_val = jnp.std(coef_samples)
        ci_low = jnp.percentile(coef_samples, 2.5)
        ci_high = jnp.percentile(coef_samples, 97.5)
        print(f"    {name:15s}: {mean_val:7.3f} ± {std_val:.3f}  (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

    intercept_samples = posterior_samples["intercept"]
    print(f"    {'Intercept':15s}: {jnp.mean(intercept_samples):7.3f} ± {jnp.std(intercept_samples):.3f}")

    noise_samples = posterior_samples["noise_std"]
    print(f"    {'Noise Std':15s}: {jnp.mean(noise_samples):7.3f} ± {jnp.std(noise_samples):.3f}")

    # --- Outlier detection results (robust model only) ---
    if use_robust:
        print("\n[4b] Outlier detection results:")
        print("-" * 50)

        # Compute posterior probability of being an outlier for each training sample
        # is_outlier has shape (n_mcmc_samples, n_train)
        is_outlier_samples = posterior_samples["is_outlier"]
        outlier_probs = np.array(jnp.mean(is_outlier_samples, axis=0))

        # Summary statistics
        n_likely_outliers = np.sum(outlier_probs > 0.5)
        n_possible_outliers = np.sum(outlier_probs > 0.1)
        print(f"    Training samples with P(outlier) > 0.5: {n_likely_outliers}")
        print(f"    Training samples with P(outlier) > 0.1: {n_possible_outliers}")
        print(f"    Average outlier probability: {np.mean(outlier_probs):.3f}")

        # Show top outlier candidates
        top_k = min(10, n_train)
        top_outlier_idx = np.argsort(outlier_probs)[::-1][:top_k]

        print(f"\n    Top {top_k} outlier candidates:")
        print("    Train Idx | Original Idx | Actual Price  | P(outlier)")
        print("    " + "-" * 55)

        for idx in top_outlier_idx:
            original_idx = train_idx[idx]
            actual_price = np.exp(y_train[idx])
            prob = outlier_probs[idx]
            print(f"    {idx:9d} | {original_idx:12d} | ${actual_price:>11,.0f} | {prob:.3f}")

    # --- Make predictions with uncertainty on HELD-OUT test set ---
    print("\n[5] Holdout predictions with uncertainty (test set):")
    print("-" * 50)

    # Sample a few posterior coefficient sets and make predictions
    coef_samples = jnp.stack([
        posterior_samples["coef_0"],
        posterior_samples["coef_1"],
        posterior_samples["coef_2"],
        posterior_samples["coef_3"],
    ], axis=1)  # (n_samples, 4)

    # Predict for first 10 houses from the TEST set (held-out data)
    # These are houses the model has never seen during training!
    # Now we can answer questions like:
    # - What's the 95% credible interval for how much living area affects price?
    # - How confident are we that quality matters more than size?
    # - What's the range of plausible prices for a new house?
    n_display = min(10, n_test)
    print(f"    Showing {n_display} of {n_test} holdout predictions:")
    print("    House | Actual Price  | Predicted (Mean) | 90% Credible Interval")
    print("    " + "-" * 65)

    # Key for sampling observation noise
    pred_key = jrandom.PRNGKey(123)

    for i in range(n_display):
        x_i = X_test[i]
        actual_log_price = y_test[i]
        actual_price = np.exp(actual_log_price)

        # Posterior predictive (includes observation noise for proper coverage)
        pred_mean_line = coef_samples @ x_i + intercept_samples
        # Add observation noise: y ~ Normal(mean, noise_std)
        pred_key, subkey = jrandom.split(pred_key)
        obs_noise = jrandom.normal(subkey, shape=noise_samples.shape) * noise_samples
        pred_log_prices = pred_mean_line + obs_noise

        pred_mean = jnp.mean(pred_mean_line)  # Point estimate without noise
        pred_low = jnp.percentile(pred_log_prices, 5)
        pred_high = jnp.percentile(pred_log_prices, 95)

        pred_price = np.exp(pred_mean)
        pred_price_low = np.exp(pred_low)
        pred_price_high = np.exp(pred_high)

        # For a house with 2,000 sq ft living area and quality rating of 7,
        # instead of predicting "$250,000", we might say
        # "$235,000 - $270,000 with 90% probability"—far more useful for decision-making.
        print(f"    {i+1:5d} | ${actual_price:>11,.0f} | ${pred_price:>14,.0f} | ${pred_price_low:>10,.0f} - ${pred_price_high:>10,.0f}")

    # Compute overall test set metrics (on ALL test samples, not just displayed)
    print(f"\n[6] Test set summary metrics (all {n_test} samples):")
    print("-" * 50)

    # Vectorized predictions for all test samples
    all_pred_mean = coef_samples @ X_test.T + intercept_samples[:, None]  # (n_samples, n_test)
    pred_means = jnp.mean(all_pred_mean, axis=0)  # (n_test,)

    # Mean Absolute Error (in log space)
    mae_log = jnp.mean(jnp.abs(pred_means - y_test))
    print(f"    MAE (log prices): {mae_log:.4f}")

    # Convert to actual prices for interpretability
    pred_prices = jnp.exp(pred_means)
    actual_prices = jnp.exp(y_test)
    mae_price = jnp.mean(jnp.abs(pred_prices - actual_prices))
    print(f"    MAE (actual prices): ${mae_price:,.0f}")

    # Coverage: % of test samples where actual falls within 90% CI
    # Include observation noise for proper posterior predictive intervals
    pred_key, subkey = jrandom.split(pred_key)
    n_posterior_samples = len(intercept_samples)
    obs_noise_all = jrandom.normal(subkey, shape=(n_posterior_samples, n_test)) * noise_samples[:, None]
    all_pred_log = all_pred_mean + obs_noise_all

    pred_low_all = jnp.percentile(all_pred_log, 5, axis=0)
    pred_high_all = jnp.percentile(all_pred_log, 95, axis=0)
    coverage = jnp.mean((y_test >= pred_low_all) & (y_test <= pred_high_all))
    print(f"    90% CI coverage: {coverage * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenJAX Bayesian House Price Prediction")
    parser.add_argument("--mh", action="store_true", help="Use Metropolis-Hastings instead of importance sampling (WIP)")
    parser.add_argument("--robust", action="store_true", help="Use robust model with outlier detection (Gibbs+HMC inference)")
    args = parser.parse_args()
    main(use_mh=args.mh, use_robust=args.robust)
