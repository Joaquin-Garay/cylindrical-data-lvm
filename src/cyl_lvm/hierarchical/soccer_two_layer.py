"""Soccer-specific wrapper for hierarchical two-layer models."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .two_layer import TwoLayerMoM
from ..distributions import MultivariateGaussian, VonMises


class SoccerTwoLayerMoM:
    """Add soccer-only plotting utilities to a two-layer MoM instance."""

    def __init__(self, model: TwoLayerMoM):
        if not isinstance(model, TwoLayerMoM):
            raise TypeError("model must be TwoLayerMoM or a subclass.")
        self._model = model

    @property
    def model(self) -> TwoLayerMoM:
        """Return the wrapped two-layer model."""
        return self._model

    def __getattr__(self, name: str) -> Any:
        # Forward all unknown attributes/methods to wrapped model
        return getattr(self._model, name)

    def plot(
        self,
        *,
        figsize: float = 6,
        arrow_scale: float = 12.0,
        title: str = "",
        show_title: bool = False,
        save: bool = False,
        file_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot first-layer Gaussian components and second-layer Von Mises means.

        This visualization is available only when layer-1 components are
        `MultivariateGaussian` and layer-2 components are `VonMises`.
        Each layer-1 component is drawn as an ellipse, and each associated
        layer-2 component is drawn as an arrow. Arrow length is proportional
        to Von Mises mean resultant length.
        """
        try:
            import matplotsoccer as mps
            import matplotlib.pyplot as plt
            from ..utils import add_arrow, add_ellips
        except ModuleNotFoundError as exc:
            raise ImportError(
                "SoccerTwoLayerMoM.plot requires optional dependencies "
                "'matplotsoccer' and 'matplotlib'."
            ) from exc

        plot_cond = (
            isinstance(self._model.layer1_mixture.components[0], MultivariateGaussian)
            and isinstance(self._model.layer2_mixtures[0].components[0], VonMises)
        )
        if not plot_cond:
            raise ValueError("Plot only available for MultivariateGaussian -> VonMises Mixture-of-mixtures.")

        ax = mps.field(show=False, figsize=figsize)
        cmap = plt.cm.plasma

        for l1_idx, (layer1_component, layer2_mixture) in enumerate(
            zip(self._model.layer1_mixture.components, self._model.layer2_mixtures)
        ):
            prior = self._model.layer1_mixture.weights[l1_idx]
            col = cmap(-0.8 * prior + 0.9)
            mean, cov = layer1_component.params
            add_ellips(ax, mean, cov, color=col, alpha=0.5)
            x0, y0 = mean

            for layer2_component in layer2_mixture.components:
                angle_mean, _ = layer2_component.params
                r = layer2_component.mean_length
                length = arrow_scale * r
                dx, dy = np.cos(angle_mean), np.sin(angle_mean)
                add_arrow(ax, x0, y0,
                          length * dx, length * dy,
                          linewidth=0.8)

        if show_title:
            plt.title(title)
        if save:
            if not file_name:
                raise ValueError("file_name must be provided when save=True.")
            plt.savefig(f"{file_name}.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
