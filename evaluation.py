import os
import argparse
import numpy as np
import torch
import datetime

from conditional_evaluation import ConditionalEvaluation
from fd import (
    compute_FD_with_reps,
    compute_statistics,
    compute_FD_with_stats,
    compute_statistics_weighted,
)
from prdc import compute_prdc


VENDI_MAX_SAMPLES = 15000

def empty_cuda_cache(device):
    if isinstance(device, torch.device):
        is_cuda = device.type == "cuda"
    else:
        is_cuda = str(device).startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()


def basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_features(path, key="dino_features"):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")
        return data[key]
    raise ValueError(f"Unsupported feature file format: {path}")


def load_weights_file(path, indices_key="indices", weights_key="weights"):
    data = np.load(path)

    if indices_key not in data:
        raise KeyError(f"Key '{indices_key}' not found in {path}. Available keys: {list(data.keys())}")
    if weights_key not in data:
        raise KeyError(f"Key '{weights_key}' not found in {path}. Available keys: {list(data.keys())}")

    indices = data[indices_key]
    weights = np.asarray(data[weights_key], dtype=np.float64)
    weights = np.clip(weights, 0, None)

    s = weights.sum()
    if s <= 0:
        raise ValueError("Weights sum must be positive.")
    weights = weights / s

    return indices, weights


def flatten_metrics(metrics):
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = float(sub_v)
        else:
            flat[k] = float(v)
    return flat


def make_scores_filename(fake_path, real_path, n_samples, has_weights):
    fake_name = basename_no_ext(fake_path)
    real_name = basename_no_ext(real_path)
    mode = "uniform_weighted" if has_weights else "uniform"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    return f"{fake_name}__vs__{real_name}__n{n_samples}__{mode}_{timestamp}.txt"


def save_scores_txt(path, sections, fake_path, real_path, n_samples, has_weights):
    fake_name = basename_no_ext(fake_path)
    real_name = basename_no_ext(real_path)

    with open(path, "w") as f:
        f.write(f"fake_features: {fake_name}\n")
        f.write(f"real_features: {real_name}\n")
        f.write(f"num_samples: {n_samples}\n")
        f.write(f"includes_weighted: {has_weights}\n")
        f.write("=" * 60 + "\n\n")

        for title, metrics in sections:
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            flat = flatten_metrics(metrics)
            for k, v in flat.items():
                f.write(f"{k.upper():<24s}: {v:.6f}\n")
            f.write("\n")


def select_indices(n_available, n_select, use_first_samples=False, seed=42):
    n_select = min(n_select, n_available)
    if use_first_samples:
        return np.arange(n_select)
    rng = np.random.default_rng(seed)
    return rng.choice(n_available, size=n_select, replace=False)


def maybe_subsample(feats, max_samples=VENDI_MAX_SAMPLES, use_first=False, seed=42):
    if max_samples is None or len(feats) <= max_samples:
        return feats
    if use_first:
        return feats[:max_samples]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=max_samples, replace=False)
    return feats[idx]


def weighted_subsample(feats, weights, max_samples=VENDI_MAX_SAMPLES, use_first=False, seed=42):
    if max_samples is None or len(feats) <= max_samples:
        return feats, weights

    if use_first:
        idx = np.arange(max_samples)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), size=max_samples, replace=False, p=weights)

    sub_feats = feats[idx]
    sub_weights = np.asarray(weights[idx], dtype=np.float64)
    sub_weights = np.clip(sub_weights, 0, None)
    sub_weights /= sub_weights.sum()
    return sub_feats, sub_weights


def resample_with_weights(fake_feats, weights, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fake_feats), size=len(fake_feats), replace=True, p=weights)
    return fake_feats[idx]


@torch.no_grad()
def compute_mmd_torch(
    feat_real,
    feat_gen,
    n_subsets=1,
    subset_size=10000,
    degree=3,
    gamma=None,
    coef0=1.0,
    device="cuda",
):
    feat_real = torch.as_tensor(feat_real, dtype=torch.float32, device=device)
    feat_gen = torch.as_tensor(feat_gen, dtype=torch.float32, device=device)

    m = min(feat_real.shape[0], feat_gen.shape[0])
    subset_size = min(subset_size, m)

    if gamma is None:
        gamma = 1.0 / feat_real.shape[1]

    mmds = np.zeros(n_subsets, dtype=np.float64)

    for i in range(n_subsets):
        idx_r = torch.randperm(feat_real.shape[0], device=device)[:subset_size]
        idx_g = torch.randperm(feat_gen.shape[0], device=device)[:subset_size]

        X = feat_real[idx_r]
        Y = feat_gen[idx_g]

        K_XX = (gamma * (X @ X.T) + coef0) ** degree
        K_YY = (gamma * (Y @ Y.T) + coef0) ** degree
        K_XY = (gamma * (X @ Y.T) + coef0) ** degree

        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

        Kt_XX_sum = K_XX.sum() - diag_X.sum()
        Kt_YY_sum = K_YY.sum() - diag_Y.sum()
        K_XY_sum = K_XY.sum()

        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (subset_size * (subset_size - 1))
        mmd2 -= 2 * K_XY_sum / (subset_size * subset_size)
        mmds[i] = float(mmd2.item())

        del X, Y, K_XX, K_YY, K_XY, diag_X, diag_Y
        empty_cuda_cache(device)

    del feat_real, feat_gen
    empty_cuda_cache(device)
    return mmds


def evaluate_uniform(
    real_feats,
    fake_feats,
    device="cuda",
    image_sigma=35.0,
    text_sigma=0.6,
    mmd_subset_size=10000,
    nearest_k=5,
    use_first_samples=False,
    seed=42,
):
    metrics = {}

    vendi_fake = maybe_subsample(
        fake_feats,
        max_samples=VENDI_MAX_SAMPLES,
        use_first=use_first_samples,
        seed=seed,
    )

    for order in [1, 2]:
        eval_model = ConditionalEvaluation(sigma=(image_sigma, text_sigma))
        ent = eval_model.compute_entropy(vendi_fake, order=order).detach().cpu()
        metrics[f"vendi_{order}"] = float(np.exp(ent).item())
        del eval_model, ent
        empty_cuda_cache(device)

    metrics["kd"] = float(
        compute_mmd_torch(
            real_feats,
            fake_feats,
            n_subsets=1,
            subset_size=mmd_subset_size,
            device=device,
        )[0]
    )
    empty_cuda_cache(device)

    metrics["fd"] = float(compute_FD_with_reps(real_feats, fake_feats))
    empty_cuda_cache(device)

    # metrics["prdc"] = compute_prdc(
    #     real_features=real_feats,
    #     fake_features=fake_feats,
    #     nearest_k=nearest_k,
    # )
    # empty_cuda_cache(device)

    return metrics


def evaluate_weighted(
    real_feats,
    fake_feats,
    weights,
    device="cuda",
    image_sigma=35.0,
    text_sigma=0.6,
    mmd_subset_size=10000,
    nearest_k=5,
    use_first_samples=False,
    seed=42,
):
    metrics = {}

    vendi_fake, vendi_weights = weighted_subsample(
        fake_feats,
        weights,
        max_samples=VENDI_MAX_SAMPLES,
        use_first=use_first_samples,
        seed=seed,
    )

    resampled_vendi = resample_with_weights(vendi_fake, vendi_weights, seed=seed)
    resampled_fake = resample_with_weights(fake_feats, weights, seed=seed)
    del vendi_fake
    empty_cuda_cache(device)

    for order in [1, 2]:
        eval_model = ConditionalEvaluation(sigma=(image_sigma, text_sigma))
        ent = eval_model.compute_entropy(resampled_vendi, order=order).detach().cpu()
        metrics[f"vendi_{order}"] = float(np.exp(ent).item())
        del eval_model, ent
        empty_cuda_cache(device)

    metrics["kd"] = float(
        compute_mmd_torch(
            real_feats,
            resampled_fake,
            n_subsets=1,
            subset_size=mmd_subset_size,
            device=device,
        )[0]
    )
    empty_cuda_cache(device)
    print('oh im here')
    mu1, sigma1 = compute_statistics(real_feats)
    mu2, sigma2 = compute_statistics_weighted(fake_feats, weights=weights)
    metrics["fd"] = float(compute_FD_with_stats(mu1, mu2, sigma1, sigma2, eps=1e-6))
    empty_cuda_cache(device)
    print('im here')

    # metrics["prdc"] = compute_prdc(
    #     real_features=real_feats,
    #     fake_features=resampled_fake,
    #     nearest_k=nearest_k,
    # )
    # empty_cuda_cache(device)

    del vendi_weights, resampled_vendi, resampled_fake
    empty_cuda_cache(device)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate uniform and weighted generative-model feature scores")

    parser.add_argument("--real_features", type=str, required=True, help="Path to real feature file (.npz or .npy)")
    parser.add_argument("--fake_features", type=str, required=True, help="Path to fake feature file (.npz or .npy)")
    parser.add_argument("--weights_file", type=str, default=None, help="Path to combined_weights.npz")
    parser.add_argument('--weighted_only', action='store_true')

    parser.add_argument("--real_key", type=str, default="dino_features", help="Key for real features in npz")
    parser.add_argument("--fake_key", type=str, default="dino_features", help="Key for fake features in npz")
    parser.add_argument("--indices_key", type=str, default="indices", help="Key for indices in weights npz")
    parser.add_argument("--weights_key", type=str, default="weights", help="Key for weights in weights npz")

    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples if no weights file is provided")
    parser.add_argument("--use_first_samples", action="store_true", help="Use first N samples instead of random sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random selection and weighted resampling")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_sigma", type=float, default=35.0)
    parser.add_argument("--text_sigma", type=float, default=0.6)
    parser.add_argument("--mmd_subset_size", type=int, default=10000)
    parser.add_argument("--nearest_k", type=int, default=5)

    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output txt file")

    args = parser.parse_args()

    real_data = load_features(args.real_features, key=args.real_key)
    fake_data = load_features(args.fake_features, key=args.fake_key)

    if args.weights_file is not None:
        indices, weights = load_weights_file(
            args.weights_file,
            indices_key=args.indices_key,
            weights_key=args.weights_key,
        )

        fake_eval = fake_data[indices]
        n_eval = len(fake_eval)

        if len(weights) != n_eval:
            raise ValueError(
                f"Mismatch: len(weights)={len(weights)} but len(selected fake features)={n_eval}"
            )

        if args.use_first_samples:
            real_indices = np.arange(min(n_eval, len(real_data)))
        else:
            real_indices = select_indices(
                len(real_data),
                n_eval,
                use_first_samples=False,
                seed=args.seed,
            )

        real_eval = real_data[real_indices]
    else:
        if args.num_samples is None:
            n_eval = min(len(real_data), len(fake_data))
        else:
            n_eval = min(args.num_samples, len(real_data), len(fake_data))

        real_indices = select_indices(
            len(real_data),
            n_eval,
            use_first_samples=args.use_first_samples,
            seed=args.seed,
        )
        fake_indices = select_indices(
            len(fake_data),
            n_eval,
            use_first_samples=args.use_first_samples,
            seed=args.seed + 1,
        )

        real_eval = real_data[real_indices]
        fake_eval = fake_data[fake_indices]
        weights = None

    print(f"Num real: {len(real_eval)} Num fake: {len(fake_eval)}")
    print(f"Vendi will use at most {VENDI_MAX_SAMPLES} fake samples.")

    sections = []
    if args.weighted_only is False:
        print("Evaluating Uniform Metrics...")
        metrics_uniform = evaluate_uniform(
            real_eval,
            fake_eval,
            device=args.device,
            image_sigma=args.image_sigma,
            text_sigma=args.text_sigma,
            mmd_subset_size=args.mmd_subset_size,
            nearest_k=args.nearest_k,
            use_first_samples=args.use_first_samples,
            seed=args.seed,
        )

        sections.append(("UNIFORM", metrics_uniform))
    print(sections)
    torch.cuda.empty_cache()

    metrics_weighted = None
    if args.weights_file is not None:
        print("Evaluating Weighted Metrics...")
        metrics_weighted = evaluate_weighted(
            real_eval,
            fake_eval,
            weights,
            device=args.device,
            image_sigma=args.image_sigma,
            text_sigma=args.text_sigma,
            mmd_subset_size=args.mmd_subset_size,
            nearest_k=args.nearest_k,
            use_first_samples=args.use_first_samples,
            seed=args.seed,
        )
        sections.append(("WEIGHTED", metrics_weighted))

    if args.output_dir is None:
        if args.weights_file is not None:
            output_dir = os.path.dirname(os.path.abspath(args.weights_file))
        else:
            output_dir = os.getcwd()
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    output_name = make_scores_filename(
        args.fake_features,
        args.real_features,
        len(fake_eval),
        args.weights_file is not None,
    )
    scores_path = os.path.join(output_dir, output_name)

    save_scores_txt(
        scores_path,
        sections,
        fake_path=args.fake_features,
        real_path=args.real_features,
        n_samples=len(fake_eval),
        has_weights=(args.weights_file is not None),
    )

    print(f"Scores saved to: {scores_path}")
    print()

    if args.weighted_only is False:
        print("=== UNIFORM ===")
        for k, v in flatten_metrics(metrics_uniform).items():
            print(f"{k}: {v:.6f}")

    if metrics_weighted is not None:
        print()
        print("=== WEIGHTED ===")
        for k, v in flatten_metrics(metrics_weighted).items():
            print(f"{k}: {v:.6f}")

    empty_cuda_cache(args.device)


if __name__ == "__main__":
    main()
