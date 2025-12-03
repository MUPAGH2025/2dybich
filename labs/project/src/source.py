from itertools import cycle
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pint
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import dtype, float64, int64, ndarray
from open_atmos_jupyter_utils import show_anim
from PyMPDATA import (
    Options,
    ScalarField,
    Solver,
    Stepper,
    VectorField,
    boundary_conditions,
)
from PyMPDATA.scalar_field import ScalarField
from PyMPDATA.solver import Solver
from scipy.signal import find_peaks


def bath_no_bath(X: ndarray, Y: ndarray):
    return np.zeros_like(X)


def bath_atan_coast(X: ndarray, Y: ndarray, x0: float = 0.2, scale: float = 30):
    return np.arctan(scale * (X - x0)) / np.pi + 0.5


def bath_power(X: ndarray, Y: ndarray, scale: float = 0.1, p: float = 0.8):
    return scale * X**p


def bath_log(X: ndarray, Y: ndarray, scale: float = 0.5, b: float = 0.1):
    return scale * np.log(b * X + 1)


def bath_power_step(
    X: np.ndarray,
    Y: np.ndarray,
    scale: float = 1,
    p: float = 0.7,
    A: float = 0.6,
    periods: int = 1,
    x0: float = 0,
    x1: float = 1,
):
    f = scale * X**p
    mask = (X >= x0) & (X <= x1)
    X_new = X[mask] - x0

    if len(X_new) > 0:
        X_max = np.max(X_new)
        k = periods / X_max if X_max != 0 else 0

        f[mask] -= A * (1 - np.cos(2 * np.pi * k * X_new))

    return f


def bath_linear_step(
    X: np.ndarray,
    Y: np.ndarray,
    scale: float = 1,
    b: float = 0,
    A: float = 0.6,
    periods: int = 1,
    x0: float = 0,
    x1: float = 1,
):
    f = scale * X + b
    mask = (X >= x0) & (X <= x1)
    X_new = X[mask] - x0

    if len(X_new) > 0:
        X_max = np.max(X_new)
        k = periods / X_max if X_max != 0 else 0

        f[mask] -= A * (1 - np.cos(2 * np.pi * k * X_new))

    return f


def bath_linear(
    X: np.ndarray,
    Y: np.ndarray,
    scale: float = 1,
    b: float = 0,
):
    f = scale * X + b
    return f


def bath_step(
    X: ndarray,
    Y: ndarray,
    A: float = 0.6,
    periods: int = 1,
    x0: float = 0,
    x1: float = 1,
):
    f = np.zeros_like(X)

    mask = (X >= x0) & (X <= x1)
    X_new = X[mask] - x0

    if len(X_new) > 0:
        X_max = np.max(X_new)
        k = periods / X_max if X_max != 0 else 0
        f[mask] = -A * (1 - np.cos(2 * np.pi * k * X_new))
    return f


def bath_parabolic_coast(
    X: ndarray,
    Y: ndarray,
    a: float = 1,
    x0: float = 0,
    y0: float = 0,
    A: float = 1,
    sigma: tuple[float, float] = (0.02, 0.02),
    xlim: tuple[float, float] = (0, 1),
    ylim: tuple[float, float] = (0, 1),
    N: int = 200,
):
    t = np.linspace(xlim[0], xlim[1], N)
    x = t
    y = a * (t - x0) ** 2 + y0

    mask_y = (y >= ylim[0]) & (y <= ylim[1])
    x = x[mask_y]
    y = y[mask_y]

    Z = np.zeros_like(X)

    for xi, yi in zip(x, y):
        G = A * (
            np.exp(
                -((X - xi) ** 2) / (2 * sigma[0] ** 2)
                - ((Y - yi) ** 2) / (2 * sigma[1] ** 2)
            )
        )
        Z = np.maximum(Z, G)

    return -Z


class Plots:
    def __init__(
        self,
        X: ndarray,
        Y: ndarray,
        bath: ndarray,
        h_init: ndarray,
        psi: ndarray,
        mag: ndarray,
        ux: ndarray,
        uy: ndarray,
        frames: ndarray,
    ) -> None:
        self.X = X
        self.Y = Y
        self.bath = bath
        self.h_init = h_init
        self.psi = psi
        self.mag = mag
        self.ux = ux
        self.uy = uy
        self.frames = frames

    @property
    def scalar_formatter(self) -> ScalarFormatter:
        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        return formatter

    def plot_bath_1d(
        self,
        ax: Axes,
        X: ndarray | None = None,
        bath: ndarray | None = None,
    ):
        X = self.X if X is None else X
        bath = self.bath if bath is None else bath

        bath = -bath[:, 0]
        ax.plot(X[:, X.shape[1] // 2], bath, c="gray", label=r"$\mathrm{bathymetry}$")
        ax.axhline(0, ls="--", c="crimson")
        ax.legend()

        ax.xaxis.set_major_formatter(self.scalar_formatter)
        ax.yaxis.set_major_formatter(self.scalar_formatter)
        ax.set_xlabel(r"$\mathrm{x}$")
        ax.set_ylabel(r"$\mathrm{z}$")
        ax.grid(alpha=0.6, ls="--")
        ylim = (bath.min(), max(bath.max(), 0))
        amp = ylim[1] - ylim[0]
        ax.set_ylim(top=ylim[1] + 0.2 * amp, bottom=ylim[0] - 0.2 * amp)

    def axvline_t(self, ax: Axes, ts: list[int], label: str = "t", **kwargs):
        for i, t in enumerate(ts):
            ax.axvline(t, **kwargs)
            ax.text(
                t,
                self.X.max() * 0.9,
                r"$\mathrm{" + label + f"_{i}" + "}$",
                fontsize=12,
                c="k",
                bbox=dict(
                    facecolor="w",
                    alpha=1,
                    edgecolor="black",
                    boxstyle="round,pad=0.3",
                ),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    def plot_time(
        self,
        ax: Axes,
        data: np.ndarray,
        X: np.ndarray | None = None,
        label: str = "",
        xlabel: str = "x",
        invert_ax: bool = True,
    ):
        t = range(data.shape[0])

        if X is None:
            x = range(data.shape[1])
        else:
            x = X

        mesh = np.meshgrid(x, t)
        vlim = (np.min(data), np.max(data))
        kwargs = {"origin": "lower", "vmin": vlim[0], "vmax": vlim[1]}
        if invert_ax:
            cont = ax.contourf(mesh[1], mesh[0], data, **kwargs)
            ax.set_xlabel(r"$\mathrm{t}$")
            ax.set_ylabel(r"$\mathrm{" + xlabel + "}$")
        else:
            cont = ax.contourf(mesh[0], mesh[1], data, **kwargs)
            ax.set_xlabel(r"$\mathrm{" + xlabel + "}$")
            ax.set_ylabel(r"$\mathrm{t}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cont, cax=cax)
        cbar.set_label(r"$\mathrm{" + label + "}$")
        cbar.ax.yaxis.set_major_formatter(self.scalar_formatter)
        ax.xaxis.set_major_formatter(self.scalar_formatter)
        ax.yaxis.set_major_formatter(self.scalar_formatter)

    def plot_bath_times(
        self,
        X: ndarray | None = None,
        Y: ndarray | None = None,
        bath: ndarray | None = None,
        psi: ndarray | None = None,
        mag: ndarray | None = None,
        ux: ndarray | None = None,
        uy: ndarray | None = None,
    ):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        bath = self.bath[:, X.shape[1] // 2] if bath is None else bath
        psi = self.psi[:, :, X.shape[1] // 2] if psi is None else psi
        mag = self.mag[:, :, X.shape[1] // 2] if mag is None else mag
        ux = self.ux[:, :, X.shape[1] // 2] if ux is None else ux
        fig, axes = plt.subplot_mosaic(
            [
                ["ux", "zeta", "magnitude"],
                ["ux", "zeta", "magnitude"],
                ["bathymetry", "bathymetry", "bathymetry"],
            ],
            constrained_layout=True,
            figsize=(16, 8),
        )
        fig.set_constrained_layout_pads(  # type: ignore
            w_pad=0.25,
        )
        self.plot_time(
            axes["ux"], ux, X[:, X.shape[1] // 2], invert_ax=True, label=r"\upsilon_x"
        )
        self.plot_time(
            axes["magnitude"],
            mag,
            X[:, X.shape[1] // 2],
            invert_ax=True,
            label=r"|\upsilon|",
        )
        self.plot_time(
            axes["zeta"], psi, X[:, X.shape[1] // 2], invert_ax=True, label=r"\zeta_x"
        )
        self.plot_bath_1d(axes["bathymetry"])
        for ax, l in zip(axes.values(), cycle("abcdef")):
            ax.set_title(r"$\mathrm{(" + l + ")}$")
        return fig, axes

    def plot_initial_3d(
        self,
        fig: Figure,
        X: ndarray | None = None,
        Y: ndarray | None = None,
        bath: ndarray | None = None,
        h_init: ndarray | None = None,
    ):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        bath = self.bath if bath is None else bath
        h_init = self.h_init if h_init is None else h_init
        # Plot
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        surf_bath = ax.plot_surface(
            X,
            Y,
            -bath,
            cmap="gnuplot",
            cstride=1,
            rstride=1,
            antialiased=1,
            alpha=0.7,
        )
        surf_h = ax.plot_surface(
            X,
            Y,
            h_init - bath,
            cmap="viridis",
            cstride=1,
            rstride=1,
            antialiased=1,
            alpha=0.7,
        )

        # Labels
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

        # Colorbar
        cbar_bath = plt.colorbar(surf_bath, ax=ax, shrink=0.6, pad=0.05)
        cbar_h = plt.colorbar(surf_h, ax=ax, shrink=0.6, pad=0.1)
        cbar_bath.set_label(r"$\mathrm{Bathymetry}$", fontsize=18)
        cbar_h.set_label(r"$\mathrm{H_{init}}$", fontsize=18)

        ax.xaxis.set_major_formatter(self.scalar_formatter)
        ax.yaxis.set_major_formatter(self.scalar_formatter)
        cbar_bath.ax.yaxis.set_major_formatter(self.scalar_formatter)
        cbar_h.ax.yaxis.set_major_formatter(self.scalar_formatter)
        # ax.view_init(azim=45)
        return ax

    def plot_pcolormesh(
        self,
        ax: Axes,
        Z: ndarray,
        X: ndarray | None = None,
        Y: ndarray | None = None,
        label: str = "",
    ):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        pcm = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="nearest")
        ax.set_aspect("equal", adjustable="box")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(pcm, cax=cax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.xaxis.set_major_formatter(self.scalar_formatter)
        ax.yaxis.set_major_formatter(self.scalar_formatter)
        cbar.ax.yaxis.set_major_formatter(self.scalar_formatter)
        cbar.set_label(r"$\mathrm{" + label + "}$")

    def plot2d_anim(self):
        def _plot2d(frame):
            p = self.psi[frame]
            fig, ax = plt.subplots(constrained_layout=True)
            pcm = ax.pcolormesh(
                self.X, self.Y, p, vmin=np.min(self.psi), vmax=np.max(self.psi)
            )
            for axis in ("x", "y"):
                getattr(ax, f"set_{axis}label")(rf"${axis}$")
            ax.set_aspect("equal", adjustable="box")
            plt.colorbar(
                pcm,
                pad=0.02,
                aspect=30,
                fraction=0.02,
                label="bathymetry",
                location="left",
            )
            ax.set_title(rf"$t / \Delta t = {frame}$")
            return fig

        show_anim(_plot2d, self.frames, gif_file="anim2D.gif")


class ShallowWaterEquationsIntegrator:

    def __init__(
        self,
        *,
        h_initial: ndarray[tuple[int, ...], dtype[float64]],
        options: Options | None = None,
        bathymetry: ndarray[tuple[int, ...], dtype[float64]],
    ) -> None:
        """initializes the solvers for a given initial condition of `h` assuming zero momenta at t=0"""
        self.bathymetry = bathymetry
        options = options or Options(
            nonoscillatory=True,
            infinite_gauge=True,
        )
        X, Y, grid = 0, 1, h_initial.shape
        stepper = Stepper(
            options=options,
            grid=grid,
            n_threads=1,  # kiedy numba nie działa, bo nie działa wielowątkowość
        )
        kwargs = {
            "boundary_conditions": [boundary_conditions.Constant(value=0)] * len(grid),
            "halo": options.n_halo,
        }
        advectees: dict[str, ScalarField] = {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(np.zeros(grid), **kwargs),
            "vh": ScalarField(np.zeros(grid), **kwargs),
        }
        self.advector = VectorField(
            (np.zeros((grid[X] + 1, grid[Y])), np.zeros((grid[X], grid[Y] + 1))),
            **kwargs,
        )
        self.solvers: dict[str, Solver] = {
            k: Solver(stepper, v, self.advector) for k, v in advectees.items()
        }

    def __getitem__(self, key) -> ndarray[tuple[Any, ...], dtype[Any]]:
        """returns `key` advectee field of the current solver state"""
        return self.solvers[key].advectee.get()

    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy) -> None:
        """applies half of the source term in the given direction"""
        self[key][:] -= (
            0.5
            * g_times_dt_over_dxy
            * self["h"]
            * np.gradient(self["h"] - self.bathymetry, axis=axis)
        )

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy) -> None:
        """computes the Courant number component from fluid column height and momenta fields"""
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self["h"], where=mask, out=velocity)

        # using slices to ensure views (over copies)
        all = slice(None, None)
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = (
            velocity[
                (
                    (all_but_last, all),
                    (all, all_but_last),
                )[axis]
            ]
            + np.diff(velocity, axis=axis) / 2
        )
        courant_number = self.advector.get_component(axis)[
            ((all_but_first_and_last, all), (all, all_but_first_and_last))[axis]
        ]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(
        self,
        *,
        nt: int,
        g: float,
        dt_over_dxy: tuple,
        outfreq: int,
        eps: float = 1e-7,
    ) -> dict[str, list[Any]]:
        """integrates `nt` timesteps and returns a dictionary of solver states recorded every `outfreq` step[s]"""
        output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1):
            if it != 0:
                mask = self["h"] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(
                        axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy
                    )
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(
                        key=key,
                        axis=axis,
                        g_times_dt_over_dxy=g * dt_over_dxy[axis],
                    )
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(
                        key=key,
                        axis=axis,
                        g_times_dt_over_dxy=g * dt_over_dxy[axis],
                    )
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
        return output


class SI:
    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity
    ureg.formatter.default_format = "~P"


class Config:
    def __init__(
        self,
        grid: tuple[int, int] = (500, 100),
        depth: float = 0.5,
        dt_over_dxy: tuple[float, float] = (0.2, 0.05),
        nt: int = 1200,
        eps: float = 1e-7,
        g: float = 10,
        outfreq: int = 3,
        n_frames: int = 100,
    ) -> None:
        self.grid = grid
        self.depth = depth
        self.g = g
        self.nt = nt
        self.dt_over_dxy = dt_over_dxy
        self.outfreq = outfreq
        self.eps = eps
        self.n_frames = n_frames

    @property
    def x(self):
        return np.linspace(0, 1, self.grid[0])

    @property
    def y(self):
        return np.linspace(0, 1, self.grid[1])

    @property
    def X(self):
        return np.meshgrid(self.x, self.y, indexing="ij")[0]

    @property
    def Y(self):
        return np.meshgrid(self.x, self.y, indexing="ij")[1]

    @staticmethod
    def get_compared_data(
        data: pd.DataFrame, data_label: str, test_label: str = "test"
    ) -> pd.DataFrame:
        split_index = data.index.to_series().str.rsplit("_", n=1, expand=True)
        prefixes, numbers = split_index[0], split_index[1]

        mask_data = prefixes == data_label
        mask_test = prefixes == test_label

        data_rows = data.loc[mask_data]
        test_rows = data.loc[mask_test]

        if len(test_rows) == 1:
            result = data_rows / test_rows.iloc[0]
        elif len(test_rows) == len(data_rows):
            result_list = []
            for num in data_rows.index.to_series().str.rsplit("_", n=1).str[1].unique():
                data_row = data_rows.filter(like=f"_{num}", axis=0)
                test_row = test_rows.filter(like=f"_{num}", axis=0)
                if len(test_row) != 1:
                    raise ValueError(
                        f"Nie znaleziono dokładnie jednej pary dla numeru {num}"
                    )
                result_list.append(data_row / test_row.iloc[0])
            result = pd.concat(result_list)
        else:
            raise ValueError(
                "Niepoprawna liczba wierszy: liczba testowych wierszy powinna być 1 lub równa liczbie danych"
            )

        return result

    @staticmethod
    def get_selected_data(
        data: pd.DataFrame, data_label: str, test_label: str = "test"
    ):
        prefixes = data.index.to_series().str.rsplit("_", n=1, expand=True)[0]
        mask = prefixes.isin([data_label, test_label])
        return data.loc[mask]

    @property
    def bath_x_peaks(self) -> ndarray[tuple[Any, ...], dtype[float64]]:
        peaks, _ = find_peaks(-(self.bathymetry[:, 0] + self.depth))
        return self.x[peaks]

    def get_data_params(self, ts: list[int], data: pd.DataFrame, label: str):
        for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
            data.loc[f"{label}_{i}"] = [
                self.psi[t0 : t1 + 1].max(),
                self.mag_h[t0 : t1 + 1].max(),
                self.uh[t0 : t1 + 1].min(),
            ]

    def init_bathymetry(self, type: str, **kwargs):
        types = {
            "flat": lambda x, y: bath_no_bath(x, y, **kwargs),
            "atan": lambda x, y: bath_atan_coast(x, y, **kwargs),
            "power": lambda x, y: bath_power(x, y, **kwargs),
            "log": lambda x, y: bath_log(x, y, **kwargs),
            "power_step": lambda x, y: bath_power_step(x, y, **kwargs),
            "linear_step": lambda x, y: bath_linear_step(x, y, **kwargs),
            "linear": lambda x, y: bath_linear(x, y, **kwargs),
            "step": lambda x, y: bath_step(x, y, **kwargs),
            "parabolic_coast": lambda x, y: bath_parabolic_coast(x, y, **kwargs),
        }
        return types[type](self.X, self.Y)

    def init_h(self):
        h = np.zeros_like(self.X)
        x0, y0 = np.max(self.Y), np.mean(self.Y)
        sigma = 0.05
        h += np.exp(-(((self.X - x0) / (2 * sigma)) ** 2)) * 0.025
        return h

    def set_bath(
        self,
        type: Literal[
            "flat",
            "atan",
            "power",
            "log",
            "power_step",
            "linear_step",
            "linear",
            "step",
            "parabolic_coast",
        ],
        **kwargs,
    ):
        self.bathymetry: ndarray[tuple[int, int], dtype[float64]] = (
            self.init_bathymetry(type, **kwargs) + self.depth
        )
        self.h_initial: ndarray[tuple[int, int], dtype[float64]] = (
            self.init_h() + self.bathymetry
        )

    def run_sim(self):
        self._output: dict[str, list[Any]] = ShallowWaterEquationsIntegrator(
            h_initial=self.h_initial,
            bathymetry=self.bathymetry,
        )(
            nt=self.nt,
            g=self.g,
            dt_over_dxy=self.dt_over_dxy,
            outfreq=self.outfreq,
            eps=self.eps,
        )

    @property
    def output(self):
        if hasattr(self, "_output"):
            return self._output
        else:
            return {
                "h": np.zeros_like(self.X).tolist(),
                "uh": np.zeros_like(self.X[None, :]).tolist(),
                "vh": np.zeros_like(self.X[None, :]).tolist(),
            }

    @property
    def psi(self) -> ndarray[tuple[int, int, int], dtype[float64]]:
        return np.array(self.output["h"]) - self.bathymetry

    @property
    def uh(self) -> ndarray[tuple[int, int, int], dtype[float64]]:
        return np.array(self.output["uh"])

    @property
    def vh(self) -> ndarray[tuple[int, int, int], dtype[float64]]:
        return np.array(self.output["vh"])

    @property
    def mag_h(self) -> ndarray[tuple[int, int, int], dtype[float64]]:
        return np.sqrt(self.uh**2 + self.vh**2)

    @property
    def frames(self) -> ndarray[tuple[int], dtype[int64]]:
        return np.linspace(0, self.psi.shape[0] - 1, self.n_frames, dtype=int64)

    si = SI()

    @property
    def plots(self) -> Plots:
        return Plots(
            self.X,
            self.Y,
            self.bathymetry,
            self.h_initial,
            self.psi,
            self.mag_h,
            self.uh,
            self.vh,
            self.frames,
        )


cfg = Config()
