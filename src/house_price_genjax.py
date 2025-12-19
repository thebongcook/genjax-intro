"""
GenJAX House Price Prediction Example
Bayesian linear regression on the Kaggle House Prices dataset

This script demonstrates probabilistic programming with GenJAX for
uncertainty-aware house price prediction.

Usage:
    python house_price_genjax.py              # Default: importance sampling
    python house_price_genjax.py --mh         # Use Metropolis-Hastings
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from genjax import gen, normal, uniform, Target, ChoiceMap
from genjax.inference.smc import ImportanceK


# --- Model Definition ---

def run_metropolis_hastings(target, n_samples=1000, n_burnin=500, step_size=0.1, seed=42, adaptive=True, target_accept=0.25):
    """
    Run Metropolis-Hastings MCMC inference with optional adaptive step size.

    Args:
        target: GenJAX Target with model, args, and constraints
        n_samples: Number of posterior samples to collect (after burn-in)
        n_burnin: Number of burn-in iterations to discard
        step_size: Initial standard deviation for Gaussian random walk proposal
        seed: Random seed for reproducibility
        adaptive: Whether to adapt step size during burn-in (default True)
        target_accept: Target acceptance rate for adaptation (default 0.35)

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
        accepted = False
        u = jrandom.uniform(accept_key)
        if jnp.log(u) < log_alpha:
            current_choices = proposed_trace.get_choices()
            current_log_prob = proposed_log_weight
            accepted = True
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
    log_prices = normal(predictions, noise_std) @ "log_prices"

    return predictions


def plot_prior_posterior(posterior_samples, features, save_path="prior_posterior.png"):
    """
    Plot prior vs posterior distributions for all model parameters.

    Args:
        posterior_samples: Dictionary of posterior samples from inference
        features: List of feature names for coefficient labels
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Define priors (matching the model definition)
    priors = {
        "coef_0": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "coef_1": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "coef_2": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "coef_3": {"type": "normal", "loc": 0.0, "scale": 1.0},
        "intercept": {"type": "normal", "loc": 12.0, "scale": 1.0},
        "noise_std": {"type": "uniform", "low": 0.1, "high": 0.5},
    }

    param_labels = {
        "coef_0": f"β₀ ({features[0]})",
        "coef_1": f"β₁ ({features[1]})",
        "coef_2": f"β₂ ({features[2]})",
        "coef_3": f"β₃ ({features[3]})",
        "intercept": "Intercept",
        "noise_std": "Noise σ",
    }

    param_order = ["coef_0", "coef_1", "coef_2", "coef_3", "intercept", "noise_std"]

    for idx, param in enumerate(param_order):
        ax = axes[idx]
        samples = np.array(posterior_samples[param])
        prior_info = priors[param]

        # Determine x-axis range based on posterior samples (with padding)
        sample_min, sample_max = samples.min(), samples.max()
        sample_range = sample_max - sample_min
        x_min = sample_min - 0.3 * sample_range
        x_max = sample_max + 0.3 * sample_range

        # For prior, extend range if needed
        if prior_info["type"] == "normal":
            prior_min = prior_info["loc"] - 3 * prior_info["scale"]
            prior_max = prior_info["loc"] + 3 * prior_info["scale"]
        else:  # uniform
            prior_min = prior_info["low"]
            prior_max = prior_info["high"]

        x_min = min(x_min, prior_min)
        x_max = max(x_max, prior_max)
        x = np.linspace(x_min, x_max, 200)

        # Plot prior distribution
        if prior_info["type"] == "normal":
            prior_pdf = stats.norm.pdf(x, loc=prior_info["loc"], scale=prior_info["scale"])
        else:  # uniform
            prior_pdf = stats.uniform.pdf(x, loc=prior_info["low"],
                                          scale=prior_info["high"] - prior_info["low"])

        ax.plot(x, prior_pdf, 'b-', linewidth=2, label='Prior', alpha=0.7)
        ax.fill_between(x, prior_pdf, alpha=0.2, color='blue')

        # Plot posterior distribution (KDE of samples)
        kde = stats.gaussian_kde(samples)
        posterior_pdf = kde(x)
        ax.plot(x, posterior_pdf, 'r-', linewidth=2, label='Posterior', alpha=0.7)
        ax.fill_between(x, posterior_pdf, alpha=0.2, color='red')

        # Add posterior mean and 95% CI
        post_mean = np.mean(samples)
        post_ci_low = np.percentile(samples, 2.5)
        post_ci_high = np.percentile(samples, 97.5)
        ax.axvline(post_mean, color='red', linestyle='--', alpha=0.8, linewidth=1.5)

        # Styling
        ax.set_xlabel(param_labels[param], fontsize=11)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f"{param_labels[param]}\nμ={post_mean:.3f}, 95% CI=[{post_ci_low:.3f}, {post_ci_high:.3f}]",
                     fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prior vs Posterior Distributions\n(Bayesian House Price Model)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    📊 Prior/Posterior plot saved to: {save_path}")


def plot_predictions_scatter(y_true, posterior_samples, X, features, seed=42, save_path="predictions_scatter.png"):
    """
    Plot scatter plots of true values vs predictions from prior and posterior.

    Args:
        y_true: True log prices (test set)
        posterior_samples: Dictionary of posterior samples from inference
        X: Feature matrix for predictions (test set)
        features: List of feature names
        seed: Random seed for prior sampling
        save_path: Path to save the figure
    """
    np.random.seed(seed)
    n_samples = 500  # Number of samples to draw
    n_points = len(y_true)

    # --- Generate Prior Predictions ---
    # Sample from priors (matching model definition)
    prior_coef_0 = np.random.normal(0.0, 1.0, n_samples)
    prior_coef_1 = np.random.normal(0.0, 1.0, n_samples)
    prior_coef_2 = np.random.normal(0.0, 1.0, n_samples)
    prior_coef_3 = np.random.normal(0.0, 1.0, n_samples)
    prior_intercept = np.random.normal(12.0, 1.0, n_samples)
    prior_noise = np.random.uniform(0.1, 0.5, n_samples)

    prior_coeffs = np.stack([prior_coef_0, prior_coef_1, prior_coef_2, prior_coef_3], axis=1)  # (n_samples, 4)

    # Prior predictions: mean prediction for each test point
    prior_pred_mean = prior_coeffs @ X.T + prior_intercept[:, None]  # (n_samples, n_points)
    prior_pred_point = np.mean(prior_pred_mean, axis=0)  # Mean across samples
    prior_pred_std = np.std(prior_pred_mean, axis=0)

    # --- Generate Posterior Predictions ---
    post_coef_0 = np.array(posterior_samples["coef_0"])[:n_samples]
    post_coef_1 = np.array(posterior_samples["coef_1"])[:n_samples]
    post_coef_2 = np.array(posterior_samples["coef_2"])[:n_samples]
    post_coef_3 = np.array(posterior_samples["coef_3"])[:n_samples]
    post_intercept = np.array(posterior_samples["intercept"])[:n_samples]

    post_coeffs = np.stack([post_coef_0, post_coef_1, post_coef_2, post_coef_3], axis=1)

    # Posterior predictions
    post_pred_mean = post_coeffs @ X.T + post_intercept[:, None]  # (n_samples, n_points)
    post_pred_point = np.mean(post_pred_mean, axis=0)
    post_pred_std = np.std(post_pred_mean, axis=0)

    # Convert to actual prices for visualization
    y_true_price = np.exp(y_true)
    prior_pred_price = np.exp(prior_pred_point)
    post_pred_price = np.exp(post_pred_point)

    # --- Create Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Determine common axis limits
    all_prices = np.concatenate([y_true_price, prior_pred_price, post_pred_price])
    price_min = max(0, np.percentile(all_prices, 1) * 0.8)
    price_max = np.percentile(all_prices, 99) * 1.1

    # --- Prior Predictions Scatter ---
    ax1 = axes[0]
    ax1.scatter(y_true_price, prior_pred_price, alpha=0.4, s=30, c='blue', edgecolors='none')
    ax1.plot([price_min, price_max], [price_min, price_max], 'k--', linewidth=2, label='Perfect prediction')
    ax1.set_xlim(price_min, price_max)
    ax1.set_ylim(price_min, price_max)
    ax1.set_xlabel('True Price ($)', fontsize=12)
    ax1.set_ylabel('Predicted Price ($)', fontsize=12)
    ax1.set_title('Prior Predictions\n(Before seeing data)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Calculate metrics for prior
    prior_mae = np.mean(np.abs(prior_pred_price - y_true_price))
    prior_corr = np.corrcoef(y_true_price, prior_pred_price)[0, 1]
    ax1.text(0.95, 0.05, f'MAE: ${prior_mae:,.0f}\nCorr: {prior_corr:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Posterior Predictions Scatter ---
    ax2 = axes[1]
    ax2.scatter(y_true_price, post_pred_price, alpha=0.4, s=30, c='red', edgecolors='none')
    ax2.plot([price_min, price_max], [price_min, price_max], 'k--', linewidth=2, label='Perfect prediction')
    ax2.set_xlim(price_min, price_max)
    ax2.set_ylim(price_min, price_max)
    ax2.set_xlabel('True Price ($)', fontsize=12)
    ax2.set_ylabel('Predicted Price ($)', fontsize=12)
    ax2.set_title('Posterior Predictions\n(After Bayesian inference)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Calculate metrics for posterior
    post_mae = np.mean(np.abs(post_pred_price - y_true_price))
    post_corr = np.corrcoef(y_true_price, post_pred_price)[0, 1]
    ax2.text(0.95, 0.05, f'MAE: ${post_mae:,.0f}\nCorr: {post_corr:.3f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Format tick labels with dollar signs
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    plt.suptitle('True vs Predicted House Prices: Prior vs Posterior', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 Predictions scatter plot saved to: {save_path}")

    return {
        "prior_mae": prior_mae,
        "prior_corr": prior_corr,
        "post_mae": post_mae,
        "post_corr": post_corr
    }


def main(use_mh=False):
    print("=" * 60)
    print("GenJAX Bayesian House Price Prediction")
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

    # Build constraint map with observed training prices
    # Using vectorized constraints - single array for all observations
    constraints = ChoiceMap.d({"log_prices": y_train})

    # Create inference target
    target = Target(
        house_price_model,
        (X_train,),
        constraints
    )

    # --- Run inference ---
    if use_mh:
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

    # --- Plot prior vs posterior ---
    plot_prior_posterior(posterior_samples, features)
    scatter_metrics = plot_predictions_scatter(y_test, posterior_samples, X_test, features)

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
    n_samples = len(intercept_samples)
    obs_noise_all = jrandom.normal(subkey, shape=(n_samples, n_test)) * noise_samples[:, None]
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
    parser.add_argument("--mh", action="store_true", help="Use Metropolis-Hastings instead of importance sampling")
    args = parser.parse_args()
    main(use_mh=args.mh)
