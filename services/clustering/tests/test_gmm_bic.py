"""Unit tests for GMM + BIC cluster selection."""

import numpy as np
import pytest
from services.clustering.cluster import _select_n_clusters, assign_cluster_for_new_student


def _make_blobs(n_per_cluster: int, n_clusters: int, n_features: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 3
    data = np.vstack([
        centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
        for i in range(n_clusters)
    ])
    return data.astype(np.float32)


def test_selects_correct_k_for_clear_clusters():
    data = _make_blobs(n_per_cluster=30, n_clusters=4, n_features=5)
    model, k = _select_n_clusters(data, max_k=10)
    assert 3 <= k <= 6


def test_selects_min_k_for_small_dataset():
    data = _make_blobs(n_per_cluster=2, n_clusters=2, n_features=3)
    model, k = _select_n_clusters(data, max_k=15)
    assert k <= 4


def test_model_predicts_labels():
    data = _make_blobs(n_per_cluster=20, n_clusters=3, n_features=4)
    model, k = _select_n_clusters(data, max_k=8)
    labels = model.predict(data)
    assert labels.shape == (60,)
    assert set(labels) == set(range(k))


def test_assign_cluster_with_cached_centroids(monkeypatch):
    import services.clustering.cluster as mod
    centroids = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    monkeypatch.setattr(mod, "_centroids_cache", centroids)
    monkeypatch.setattr(mod, "_kc_order_cache", ["kc_a", "kc_b"])

    cluster = assign_cluster_for_new_student({"kc_a": 0.9, "kc_b": 1.1})
    assert cluster == 1

    cluster = assign_cluster_for_new_student({"kc_a": 2.1, "kc_b": 1.9})
    assert cluster == 2


def test_assign_cluster_returns_none_without_cache(monkeypatch):
    import services.clustering.cluster as mod
    monkeypatch.setattr(mod, "_centroids_cache", None)
    monkeypatch.setattr(mod, "_kc_order_cache", None)
    assert assign_cluster_for_new_student({"kc_a": 0.5}) is None
