"""Model builders and training orchestration for synthetic data experiments."""

import time

import numpy as np

import cyl_lvm as clvm


def mom_builder(dim: int, n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = clvm.MixtureModel(
        [clvm.MultivariateGaussian(dim) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.VonMisesFisher(dim) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.TwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)


def mom_iso_builder(dim: int, n_layer1_components, n_layer2_components, init_layer1, init_layer2, rng):
    layer1_mixture = clvm.MixtureModel(
        [clvm.MultivariateGaussian(dim) for _ in range(n_layer1_components)],
        init=init_layer1,
        rng=rng,
    )
    layer2_mixtures = [
        clvm.MixtureModel(
            [clvm.VonMisesFisher(dim) for _ in range(n_layer2_components)],
            init=init_layer2,
            rng=rng,
        )
        for _ in range(n_layer1_components)
    ]
    return clvm.IsolatedTwoLayerMoM(layer1_mixture=layer1_mixture, layer2_mixtures=layer2_mixtures)


def cylindrical_mixture_builder(dim: int, n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return clvm.MixtureModel(
        [clvm.Cylindrical(d_gauss=dim, d_vmf=dim) for _ in range(n_components)],
        init=init,
        rng=rng,
    )


def ind_cylindrical_mixture_builder(dim: int, n_components, init, rng):
    """Builder for 3D cylindrical mixtures used by BIC grid-search calibration."""
    return clvm.MixtureModel(
        [clvm.IndCylindrical(d_gauss=dim, d_vmf=dim) for _ in range(n_components)],
        init=init,
        rng=rng,
    )


def train_all_models(dim: int,
                     x,
                     setup_list: list,
                     *,
                     print_: bool = True,
                     return_training_times: bool = False,
                     return_em_iter: bool = False):
    """
    Train all configured models.

    When ``return_training_times`` is true, return ``(models, training_times)``,
    where each elapsed time is in seconds and aligned with the returned models.
    """

    x_gauss = x[:, :dim]
    x_vmf = x[:, dim:]
    model_list = []
    training_times = []
    n_iters = []

    for setup in setup_list:
        if setup["model_type"] == "cylmix":
            model = cylindrical_mixture_builder(
                        dim=dim,
                        n_components=setup["model_components"][0],
                        init="k-means",
                        rng=np.random.RandomState(42)
                )
            fit_args = (x,)
        elif setup["model_type"] == "indcylmix":
            model = ind_cylindrical_mixture_builder(
                        dim=dim,
                        n_components=setup["model_components"][0],
                        init="k-means",
                        rng=np.random.RandomState(42)
                )
            fit_args = (x,)
        elif setup["model_type"] == "mom":
            model = mom_builder(
                        dim=dim,
                        n_layer1_components=setup["model_components"][0],
                        n_layer2_components=setup["model_components"][1],
                        init_layer1="k-means",
                        init_layer2="k-means",
                        rng=np.random.RandomState(42)
                )
            fit_args = (x_gauss, x_vmf)
        elif setup["model_type"] == "isomom":
            model = mom_iso_builder(
                        dim=dim,
                        n_layer1_components=setup["model_components"][0],
                        n_layer2_components=setup["model_components"][1],
                        init_layer1="k-means",
                        init_layer2="k-means",
                        rng=np.random.RandomState(42)
                )
            fit_args = (x_gauss, x_vmf)
        else:
            model = None
            fit_args = ()

        if model is not None:
            start_time = time.perf_counter()
            model = model.fit(*fit_args)
            n_iter = model.n_iter
            training_time = time.perf_counter() - start_time
            if print_:
                print(f"Model: {setup['model_type']}. EM Iterations: {model.n_iter}")
            model_list.append(model)
            training_times.append(training_time)
            n_iters.append(n_iter)

    if return_training_times and return_em_iter:
        return_tuple = (model_list, training_times, n_iters)
    elif return_training_times:
        return_tuple = (model_list, training_times)
    elif return_em_iter:
        return_tuple = (model_list, n_iters)
    else:
        return_tuple = (model_list)
    return return_tuple


def _split_eval_data(X_eval, d_gauss):
    return X_eval[:, :d_gauss], X_eval[:, d_gauss:]


def predict_model(model, X_eval, d_gauss):
    """Dispatch model prediction for combined or split synthetic eval data."""
    if not hasattr(model, "predict"):
        raise AttributeError(f"{type(model).__name__} does not support prediction.")

    predict_fn = model.predict
    if isinstance(model, clvm.MixtureModel):
        return predict_fn(X_eval)

    return predict_fn(*_split_eval_data(X_eval, d_gauss))


def ari_model(model, labels, X_eval, d_gauss):
    """Adjusted Rand index for a model evaluated on synthetic experiment data."""
    return clvm.ari(labels, predict_model(model, X_eval, d_gauss))

def ari_full_model(model, labels_compact, X_eval, d_gauss):
    """Adjusted Rand index for a model evaluated on synthetic experiment data."""
    model_labels = model.predict_full(*_split_eval_data(X_eval, d_gauss))
    _, model_labels_compact = np.unique(model_labels, axis=0, return_inverse=True)
    return clvm.ari(labels_compact, model_labels_compact)

def score_model(model, X_eval, d_gauss, score_type: str = "score"):
    score_methods = {
        "score": "score",
        "avg_ll": "score",
        "gmpd": "gmpd",
        "bic": "bic_score",
        "bic_score": "bic_score",
        "aic": "aic_score",
        "aic_score": "aic_score",
    }
    if not isinstance(score_type, str):
        raise TypeError("score_type must be a string.")

    score_key = score_type.lower()
    if score_key not in score_methods:
        raise ValueError(
            "score_type must be one of: "
            f"{', '.join(sorted(score_methods))}."
        )

    method_name = score_methods[score_key]
    if not hasattr(model, method_name):
        raise AttributeError(
            f"{type(model).__name__} does not support score_type={score_type!r} "
            f"because it has no {method_name} method."
        )

    score_fn = getattr(model, method_name)
    if isinstance(model, clvm.MixtureModel):
        return score_fn(X_eval)
    return score_fn(*_split_eval_data(X_eval, d_gauss))
