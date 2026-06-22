"""Model builders and training orchestration for synthetic data experiments."""

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
                     x_noisy,
                     setup: dict):

    x_gauss = x[:, :dim]
    x_vmf = x[:, dim:]
    x_noisy_gauss = x_noisy[:, :dim]
    x_noisy_vmf = x_noisy[:, dim:]

    print(f"EM iterations")

    print(f"Cylindrical mixture")
    cylmix = cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["cylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x)
    print(f"   01 No noise: {cylmix.n_iter}")
    noisy_cylmix = cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["noisy_cylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy)
    print(f"   02 Noisy: {noisy_cylmix.n_iter}")

    print(f"Independent cylindrical mixture")
    indcylmix = ind_cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["indcylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x)
    print(f"   03 No noise: {indcylmix.n_iter}")
    noisy_indcylmix = ind_cylindrical_mixture_builder(
            dim=dim,
            n_components=setup["noisy_indcylmix_l1"],
            init="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy)
    print(f"   04 Noisy: {noisy_indcylmix.n_iter}")

    print(f"Mixture of mixtures")
    mom = mom_builder(
            dim=dim,
            n_layer1_components=setup["mom_l1"],
            n_layer2_components=setup["mom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_gauss, x_vmf)
    print(f"   05 No noise: {mom.n_iter}")
    noisy_mom = mom_builder(
            dim=dim,
            n_layer1_components=setup["noisy_mom_l1"],
            n_layer2_components=setup["noisy_mom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy_gauss, x_noisy_vmf)
    print(f"   06 Noisy: {noisy_mom.n_iter}")

    print("Isolation mixture of mixtures")
    isomom = mom_iso_builder(
            dim=dim,
            n_layer1_components=setup["isomom_l1"],
            n_layer2_components=setup["isomom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_gauss, x_vmf)
    print(f"   07 No noise: {isomom.n_iter}")
    noisy_isomom = mom_iso_builder(
            dim=dim,
            n_layer1_components=setup["noisy_isomom_l1"],
            n_layer2_components=setup["noisy_isomom_l2"],
            init_layer1="k-means",
            init_layer2="k-means",
            rng=np.random.RandomState(42)
                ).fit(x_noisy_gauss, x_noisy_vmf)
    print(f"   08 Noisy: {noisy_isomom.n_iter}")

    return [cylmix, noisy_cylmix, indcylmix, noisy_indcylmix, mom, noisy_mom, isomom, noisy_isomom]
