"""
GenJAX House Price Prediction Example
Bayesian linear regression on the Kaggle House Prices dataset

This script demonstrates probabilistic programming with GenJAX for
uncertainty-aware house price prediction.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd
import numpy as np

from genjax import gen, normal, uniform, Target, ChoiceMap
from genjax.inference.smc import ImportanceK


# --- Model Definition ---

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


def main():
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
    print("\n[3] Running importance sampling inference...")
    k_particles = 1000
    print(f"    Number of particles: {k_particles}")

    alg = ImportanceK(target, k_particles=k_particles)
    key = jrandom.PRNGKey(42)

    # Run importance sampling
    sub_keys = jrandom.split(key, k_particles)
    _, posterior_samples = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, target
    )

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

    # --- Make predictions with uncertainty on HELD-OUT test set ---
    print("\n[5] Holdout predictions with uncertainty (test set):")
    print("-" * 50)

    # Sample a few posterior coefficient sets and make predictions
    coef_samples = jnp.stack([
        posterior_samples["coef_0"],
        posterior_samples["coef_1"],
        posterior_samples["coef_2"],
        posterior_samples["coef_3"],
    ], axis=1)  # (k_particles, 4)

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

    for i in range(n_display):
        x_i = X_test[i]
        actual_log_price = y_test[i]
        actual_price = np.exp(actual_log_price)

        # Posterior predictive
        pred_log_prices = coef_samples @ x_i + intercept_samples
        pred_mean = jnp.mean(pred_log_prices)
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
    all_pred_log = coef_samples @ X_test.T + intercept_samples[:, None]  # (k_particles, n_test)
    pred_means = jnp.mean(all_pred_log, axis=0)  # (n_test,)

    # Mean Absolute Error (in log space)
    mae_log = jnp.mean(jnp.abs(pred_means - y_test))
    print(f"    MAE (log prices): {mae_log:.4f}")

    # Convert to actual prices for interpretability
    pred_prices = jnp.exp(pred_means)
    actual_prices = jnp.exp(y_test)
    mae_price = jnp.mean(jnp.abs(pred_prices - actual_prices))
    print(f"    MAE (actual prices): ${mae_price:,.0f}")

    # Coverage: % of test samples where actual falls within 90% CI
    pred_low_all = jnp.percentile(all_pred_log, 5, axis=0)
    pred_high_all = jnp.percentile(all_pred_log, 95, axis=0)
    coverage = jnp.mean((y_test >= pred_low_all) & (y_test <= pred_high_all))
    print(f"    90% CI coverage: {coverage * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
