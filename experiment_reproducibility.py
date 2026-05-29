#!/usr/bin/env python3
"""
Reproducible micro-experiment underpinning the revised paper.

Goal: demonstrate the MAEF-XAI evaluation protocol on a small, fully
reproducible synthetic-clinical setup so the paper's tables represent
COMPUTED values rather than synthesized literature numbers.

Two modalities are simulated:
  (a) Tabular EHR (sepsis-style binary classifier on synthetic vitals)
  (b) Imaging (low-resolution synthetic radiograph-style classifier)

Three explainers are evaluated:
  - SHAP-style (KernelSHAP approximation via permutation sampling)
  - LIME-style (local linear surrogate)
  - Integrated Gradients (on the imaging model)

Metrics (all in [0,1], normalized):
  - FID : faithfulness via deletion AUC, transformed to [0,1]
  - STAB: stability via Spearman rank correlation of explanations
          under bounded perturbation
  - PLAUS: agreement with a ground-truth "clinician" attribution we
          define from the synthetic data-generating process

TI is computed as a weighted geometric mean on the [0,1] inputs with
explicit worked-example. A sensitivity analysis is also produced.

Random seed is fixed.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import json

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# 1. TABULAR EHR SETUP
# ---------------------------------------------------------------------------
def make_tabular(n=2000, d=10):
    """Sepsis-style: 4 truly informative features, 6 noise features."""
    X = RNG.normal(0, 1, size=(n, d))
    true_coef = np.array([2.0, -1.5, 1.2, 0.8] + [0.0] * (d - 4))
    logits = X @ true_coef + RNG.normal(0, 0.5, n)
    y = (logits > 0).astype(int)
    return X, y, true_coef


def kernel_shap_approx(model, x, X_bg, n_samples=40):
    """KernelSHAP-style approximation via random feature coalitions.

    Vectorized: builds all sample pairs per feature in two predict_proba calls.
    """
    d = x.shape[0]
    phi = np.zeros(d)
    n_bg = len(X_bg)

    for j in range(d):
        # build n_samples coalitions excluding feature j
        masks = RNG.random((n_samples, d)) < 0.5
        masks[:, j] = False
        bg_idx = RNG.integers(0, n_bg, size=n_samples)
        X_off = np.where(masks, x, X_bg[bg_idx])
        X_on = X_off.copy()
        X_on[:, j] = x[j]
        p_off = model.predict_proba(X_off)[:, 1]
        p_on = model.predict_proba(X_on)[:, 1]
        phi[j] = np.mean(p_on - p_off)
    return phi


def lime_approx(model, x, X_bg, n_samples=150, sigma=0.5):
    """LIME-style local linear surrogate."""
    d = x.shape[0]
    X_local = x + RNG.normal(0, sigma, size=(n_samples, d))
    y_local = model.predict_proba(X_local)[:, 1]
    w = np.exp(-np.linalg.norm(X_local - x, axis=1) ** 2 / (2 * sigma ** 2))
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(X_local, y_local, sample_weight=w)
    return surrogate.coef_


def deletion_auc_tab(model, x, phi, n_steps=10):
    """Deletion AUC: progressively zero out top-k features by |phi|."""
    d = x.shape[0]
    order = np.argsort(-np.abs(phi))
    probs = []
    for k in range(d + 1):
        x_mod = x.copy()
        x_mod[order[:k]] = 0.0
        p = model.predict_proba(x_mod[None, :])[0, 1]
        probs.append(p)
    auc = np.trapezoid(probs, dx=1.0 / d)
    return auc


def fid_tab(model, X_eval, explain_fn, X_bg):
    """Composite fidelity in [0,1]: low deletion AUC => high fidelity.
    
    Returns 1 - normalized AUC where normalization is over the empirical
    range observed in the model so values are comparable across explainers.
    """
    aucs = []
    for x in X_eval:
        phi = explain_fn(model, x, X_bg)
        aucs.append(deletion_auc_tab(model, x, phi))
    aucs = np.array(aucs)
    # higher fidelity = lower deletion AUC after the model's natural baseline
    p_max = model.predict_proba(X_eval).mean(axis=0)[1]
    p_min = 0.0
    fid = 1.0 - (aucs - p_min) / (p_max - p_min + 1e-8)
    fid = np.clip(fid, 0.0, 1.0)
    return float(fid.mean()), float(fid.std())


def stab_tab(model, X_eval, explain_fn, X_bg, noise=0.1, reps=3):
    """Stability via Spearman rho of |phi| ranks under bounded jitter."""
    rhos = []
    for x in X_eval:
        phi_ref = explain_fn(model, x, X_bg)
        for _ in range(reps):
            x_pert = x + RNG.normal(0, noise, size=x.shape)
            phi_pert = explain_fn(model, x_pert, X_bg)
            rho, _ = spearmanr(np.abs(phi_ref), np.abs(phi_pert))
            if not np.isnan(rho):
                rhos.append(rho)
    rhos = (np.array(rhos) + 1) / 2.0  # map [-1,1] -> [0,1]
    return float(rhos.mean()), float(rhos.std())


def plaus_tab(X_eval, explain_fn, model, X_bg, true_coef):
    """Plausibility: Spearman rho of |phi| vs |true_coef| (ground truth)."""
    rhos = []
    truth = np.abs(true_coef)
    for x in X_eval:
        phi = explain_fn(model, x, X_bg)
        rho, _ = spearmanr(np.abs(phi), truth)
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = (np.array(rhos) + 1) / 2.0
    return float(rhos.mean()), float(rhos.std())


# ---------------------------------------------------------------------------
# 2. IMAGING SETUP (small synthetic 16x16 "radiograph-style")
# ---------------------------------------------------------------------------
def make_imaging(n=600, h=16):
    """Class 1 has a bright 4x4 square in the upper-left; class 0 doesn't."""
    X = RNG.normal(0.2, 0.1, size=(n, h * h))
    y = np.zeros(n, dtype=int)
    truth_mask = np.zeros((h, h))
    truth_mask[2:6, 2:6] = 1.0
    truth_flat = truth_mask.flatten()
    for i in range(n):
        if RNG.random() < 0.5:
            y[i] = 1
            img = X[i].reshape(h, h)
            img[2:6, 2:6] += 0.6
            X[i] = img.flatten()
    return X, y, truth_flat


def integrated_gradients_approx(model, x, baseline, n_steps=8):
    """Integrated Gradients via finite-difference (predict_proba).

    Uses vectorized batch evaluation for speed.
    """
    diffs = x - baseline
    grads = np.zeros_like(x)
    eps = 1e-3
    d = len(x)
    for alpha in np.linspace(0.0, 1.0, n_steps):
        x_a = baseline + alpha * diffs
        # Build batches of perturbed inputs
        X_plus = np.tile(x_a, (d, 1))
        X_minus = np.tile(x_a, (d, 1))
        idx = np.arange(d)
        X_plus[idx, idx] += eps
        X_minus[idx, idx] -= eps
        p_plus = model.predict_proba(X_plus)[:, 1]
        p_minus = model.predict_proba(X_minus)[:, 1]
        grads += (p_plus - p_minus) / (2 * eps)
    grads /= n_steps
    return diffs * grads


def kernel_shap_img(model, x, X_bg, n_samples=12):
    """Reduced KernelSHAP for imaging (random pixel coalitions). Vectorized."""
    d = x.shape[0]
    phi = np.zeros(d)
    sample_idx = RNG.choice(d, size=min(16, d), replace=False)
    n_bg = len(X_bg)
    for j in sample_idx:
        masks = RNG.random((n_samples, d)) < 0.5
        masks[:, j] = False
        bg_idx = RNG.integers(0, n_bg, size=n_samples)
        X_off = np.where(masks, x, X_bg[bg_idx])
        X_on = X_off.copy()
        X_on[:, j] = x[j]
        p_off = model.predict_proba(X_off)[:, 1]
        p_on = model.predict_proba(X_on)[:, 1]
        phi[j] = np.mean(p_on - p_off)
    return phi


def lime_img(model, x, X_bg, n_samples=150, sigma=0.05):
    d = x.shape[0]
    X_local = x + RNG.normal(0, sigma, size=(n_samples, d))
    X_local = np.clip(X_local, 0.0, 1.5)
    y_local = model.predict_proba(X_local)[:, 1]
    w = np.exp(-np.linalg.norm(X_local - x, axis=1) ** 2 / (2 * sigma ** 2 * d))
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(X_local, y_local, sample_weight=w)
    return surrogate.coef_


def fid_img(model, X_eval, explain_fn, X_bg):
    aucs = []
    for x in X_eval:
        phi = explain_fn(model, x, X_bg)
        d = x.shape[0]
        order = np.argsort(-np.abs(phi))
        probs = []
        for frac in np.linspace(0.0, 0.5, 11):
            k = int(frac * d)
            x_mod = x.copy()
            x_mod[order[:k]] = X_bg.mean(axis=0)[order[:k]]
            p = model.predict_proba(x_mod[None, :])[0, 1]
            probs.append(p)
        auc = np.trapezoid(probs, dx=0.05)
        aucs.append(auc)
    aucs = np.array(aucs)
    p_max = model.predict_proba(X_eval).mean(axis=0)[1]
    fid = 1.0 - aucs / (p_max + 1e-8)
    fid = np.clip(fid, 0.0, 1.0)
    return float(fid.mean()), float(fid.std())


def stab_img(model, X_eval, explain_fn, X_bg, noise=0.1, reps=2):
    rhos = []
    for x in X_eval:
        phi_ref = explain_fn(model, x, X_bg)
        for _ in range(reps):
            x_pert = x + RNG.normal(0, noise, size=x.shape)
            phi_pert = explain_fn(model, x_pert, X_bg)
            rho, _ = spearmanr(np.abs(phi_ref), np.abs(phi_pert))
            if not np.isnan(rho):
                rhos.append(rho)
    rhos = (np.array(rhos) + 1) / 2.0
    return float(rhos.mean()), float(rhos.std())


def plaus_img(X_eval, explain_fn, model, X_bg, truth_flat):
    rhos = []
    for x in X_eval:
        phi = explain_fn(model, x, X_bg)
        rho, _ = spearmanr(np.abs(phi), truth_flat)
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = (np.array(rhos) + 1) / 2.0
    return float(rhos.mean()), float(rhos.std())


# ---------------------------------------------------------------------------
# 3. TI COMPUTATION
# ---------------------------------------------------------------------------
def trustworthiness_index(fid, stab, plaus, alpha, beta, gamma):
    """TI = FID^alpha * STAB^beta * PLAUS^gamma, all inputs in [0,1]."""
    assert abs(alpha + beta + gamma - 1.0) < 1e-6
    return (fid ** alpha) * (stab ** beta) * (plaus ** gamma)


# ---------------------------------------------------------------------------
# 4. RUN THE EXPERIMENT
# ---------------------------------------------------------------------------
def main():
    results = {}

    # --- Tabular EHR ---
    print("[*] Tabular EHR experiment...")
    X_tab, y_tab, true_coef = make_tabular(n=1200, d=10)
    n_train = 900
    X_tr, X_te = X_tab[:n_train], X_tab[n_train:]
    y_tr, y_te = y_tab[:n_train], y_tab[n_train:]
    model_tab = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, random_state=7
    )
    model_tab.fit(X_tr, y_tr)
    auc_tab = roc_auc_score(y_te, model_tab.predict_proba(X_te)[:, 1])
    print(f"    tabular model AUC: {auc_tab:.3f}")

    X_eval_tab = X_te[:40]
    X_bg_tab   = X_tr[:120]

    print("    SHAP fidelity...")
    fid_shap, _   = fid_tab(model_tab, X_eval_tab, kernel_shap_approx, X_bg_tab)
    print("    SHAP stability...")
    stab_shap, _  = stab_tab(model_tab, X_eval_tab, kernel_shap_approx, X_bg_tab)
    print("    SHAP plausibility...")
    plaus_shap, _ = plaus_tab(X_eval_tab, kernel_shap_approx, model_tab, X_bg_tab, true_coef)

    print("    LIME fidelity...")
    fid_lime, _   = fid_tab(model_tab, X_eval_tab, lime_approx, X_bg_tab)
    print("    LIME stability...")
    stab_lime, _  = stab_tab(model_tab, X_eval_tab, lime_approx, X_bg_tab)
    print("    LIME plausibility...")
    plaus_lime, _ = plaus_tab(X_eval_tab, lime_approx, model_tab, X_bg_tab, true_coef)

    results["tabular"] = {
        "model_auc": auc_tab,
        "SHAP": {"FID": fid_shap, "STAB": stab_shap, "PLAUS": plaus_shap},
        "LIME": {"FID": fid_lime, "STAB": stab_lime, "PLAUS": plaus_lime},
    }

    # --- Imaging ---
    print("\n[*] Imaging experiment...")
    X_img, y_img, truth_flat = make_imaging(n=400, h=16)
    n_train = 300
    X_tr_i, X_te_i = X_img[:n_train], X_img[n_train:]
    y_tr_i, y_te_i = y_img[:n_train], y_img[n_train:]
    model_img = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=7
    )
    model_img.fit(X_tr_i, y_tr_i)
    auc_img = roc_auc_score(y_te_i, model_img.predict_proba(X_te_i)[:, 1])
    print(f"    imaging model AUC: {auc_img:.3f}")

    X_eval_img = X_te_i[:12]
    X_bg_img   = X_tr_i[:25]
    baseline_img = X_bg_img.mean(axis=0)

    print("    SHAP-img fidelity...")
    fid_shap_i, _ = fid_img(model_img, X_eval_img, kernel_shap_img, X_bg_img)
    print("    SHAP-img stability...")
    stab_shap_i, _ = stab_img(model_img, X_eval_img, kernel_shap_img, X_bg_img)
    print("    SHAP-img plausibility...")
    plaus_shap_i, _ = plaus_img(X_eval_img, kernel_shap_img, model_img, X_bg_img, truth_flat)

    print("    LIME-img fidelity...")
    fid_lime_i, _ = fid_img(model_img, X_eval_img, lime_img, X_bg_img)
    print("    LIME-img stability...")
    stab_lime_i, _ = stab_img(model_img, X_eval_img, lime_img, X_bg_img)
    print("    LIME-img plausibility...")
    plaus_lime_i, _ = plaus_img(X_eval_img, lime_img, model_img, X_bg_img, truth_flat)

    def ig_fn(model, x, X_bg):
        return integrated_gradients_approx(model, x, baseline_img, n_steps=6)

    print("    IG-img fidelity...")
    fid_ig_i, _ = fid_img(model_img, X_eval_img, ig_fn, X_bg_img)
    print("    IG-img stability...")
    stab_ig_i, _ = stab_img(model_img, X_eval_img, ig_fn, X_bg_img)
    print("    IG-img plausibility...")
    plaus_ig_i, _ = plaus_img(X_eval_img, ig_fn, model_img, X_bg_img, truth_flat)

    results["imaging"] = {
        "model_auc": auc_img,
        "SHAP": {"FID": fid_shap_i, "STAB": stab_shap_i, "PLAUS": plaus_shap_i},
        "LIME": {"FID": fid_lime_i, "STAB": stab_lime_i, "PLAUS": plaus_lime_i},
        "IG":   {"FID": fid_ig_i,   "STAB": stab_ig_i,   "PLAUS": plaus_ig_i},
    }

    # --- TI with worked example ---
    weights = {
        "tabular": (0.35, 0.40, 0.25),
        "imaging": (0.30, 0.45, 0.25),
    }
    results["weights"] = weights
    results["TI"] = {}
    for mod in ["tabular", "imaging"]:
        a, b, g = weights[mod]
        results["TI"][mod] = {}
        for method in results[mod]:
            if method == "model_auc":
                continue
            m = results[mod][method]
            ti = trustworthiness_index(m["FID"], m["STAB"], m["PLAUS"], a, b, g)
            results["TI"][mod][method] = ti

    # --- sensitivity analysis: vary alpha,beta,gamma by +-0.1 ---
    print("\n[*] Sensitivity analysis...")
    rankings_stable = True
    for mod in ["tabular", "imaging"]:
        base_a, base_b, base_g = weights[mod]
        methods = [m for m in results[mod] if m != "model_auc"]
        base_ranking = sorted(
            methods, key=lambda m: -results["TI"][mod][m]
        )
        for da in [-0.1, 0.0, 0.1]:
            for db in [-0.1, 0.0, 0.1]:
                a = base_a + da
                b = base_b + db
                g = 1.0 - a - b
                if a <= 0 or b <= 0 or g <= 0:
                    continue
                tis = {}
                for method in methods:
                    m = results[mod][method]
                    tis[method] = trustworthiness_index(
                        m["FID"], m["STAB"], m["PLAUS"], a, b, g
                    )
                ranking = sorted(methods, key=lambda mm: -tis[mm])
                if ranking != base_ranking:
                    rankings_stable = False
    results["rankings_stable_under_perturbation"] = rankings_stable

    # --- worked example computation ---
    a, b, g = weights["imaging"]
    m = results["imaging"]["SHAP"]
    worked = {
        "method": "SHAP (imaging)",
        "FID":   m["FID"],
        "STAB":  m["STAB"],
        "PLAUS": m["PLAUS"],
        "alpha": a, "beta": b, "gamma": g,
        "FID^alpha":   m["FID"] ** a,
        "STAB^beta":   m["STAB"] ** b,
        "PLAUS^gamma": m["PLAUS"] ** g,
        "TI": (m["FID"] ** a) * (m["STAB"] ** b) * (m["PLAUS"] ** g),
    }
    results["worked_example"] = worked

    with open("/home/claude/paper/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
