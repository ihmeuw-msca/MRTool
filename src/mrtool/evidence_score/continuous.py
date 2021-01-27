"""
Continous scorelator
"""
import os
from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from mrtool import MRData, MRBRT, MRBeRT
from mrtool.core.other_sampling import (extract_simple_lme_hessian,
                                        extract_simple_lme_specs)


class ContinuousScorelator:
    def __init__(self,
                 signal_model: Union[MRBRT, MRBeRT],
                 final_model: MRBRT,
                 alt_cov_names: List[str],
                 ref_cov_names: List[str],
                 exposure_quantiles: Tuple[float] = (0.15, 0.85),
                 exposure_bounds: Tuple[float] = None,
                 draw_quantiles: Tuple[float] = (0.05, 0.95),
                 num_samples: int = 1000,
                 num_points: int = 100,
                 ref_exposure: Union[str, float] = None,
                 j_shaped: bool = False,
                 name: str = 'unknown'):
        self.signal_model = signal_model
        self.final_model = final_model
        self.alt_cov_names = alt_cov_names
        self.ref_cov_names = ref_cov_names
        self.exposure_quantiles = exposure_quantiles
        self.exposure_bounds = exposure_bounds
        self.draw_quantiles = draw_quantiles
        self.num_samples = num_samples
        self.num_points = num_points
        self.ref_exposure = ref_exposure
        self.j_shaped = j_shaped
        self.name = name

        self.df = None
        self.exposure_limits = None
        self.pred_exposures = None
        self.pred = None
        self.ref_value = None

        self.se_model = {}
        self.se_model_all = {}
        self.linear_model_fill = None

        self.num_fill = 0
        self.score = None
        self.score_fill = None
        self.outer_draws = None
        self.inner_draws = None
        self.outer_draws_fill = None
        self.inner_draws_fill = None

    def process(self):
        self.process_data()

        self.detect_pub_bias()

        self.score, self.inner_draws, self.outer_draws = self.get_score(self.final_model)

        if self.has_pub_bias():
            self.adjust_pub_bias()

    def process_data(self):
        # extract data frame
        self.df = self.signal_model.data.to_df()

        # convert study_id to string
        self.df.study_id = self.df.study_id.astype(str)

        # get exposure information
        ref_exposures = self.df[self.ref_cov_names].to_numpy()
        alt_exposures = self.df[self.alt_cov_names].to_numpy()
        self.df["ref_mid"] = ref_exposures.mean(axis=1)
        self.df["alt_mid"] = alt_exposures.mean(axis=1)

        exposure_min = min(ref_exposures.min(), alt_exposures.min())
        exposure_max = max(ref_exposures.max(), alt_exposures.max())
        self.exposure_limits = (exposure_min, exposure_max)
        self.pred_exposures = np.linspace(*self.exposure_limits,
                                          self.num_points)
        if self.exposure_bounds is None:
            self.exposure_bounds = (
                np.quantile(self.df.ref_mid, self.exposure_quantiles[0]),
                np.quantile(self.df.alt_mid, self.exposure_quantiles[1])
            )

        # create temperary prediction
        ref_cov = np.repeat(self.exposure_limits[0], self.num_points)
        alt_cov = self.pred_exposures
        data = MRData(covs={**{name: ref_cov for name in self.ref_cov_names},
                            **{name: alt_cov for name in self.alt_cov_names}})
        pred = self.signal_model.predict(data)

        # extract reference exposure
        if self.ref_exposure is None:
            self.ref_exposure = self.exposure_limits[0]
            self.ref_value = pred[0]
        elif self.ref_exposure == "min":
            self.ref_exposure = alt_cov[np.argmin(pred)]
            self.ref_value = pred.min()
        else:
            self.ref_value = np.interp(self.ref_exposure, alt_cov, pred)

        # add outlier column
        if isinstance(self.signal_model, MRBRT):
            trim_weights = self.signal_model.w_soln
        else:
            trim_weights = np.vstack([
                model.w_soln for model in self.signal_model.sub_models
            ]).T.dot(self.signal_model.weights)
        self.df["oulier"] = (trim_weights <= 0.1).astype(int)

        # add signal column
        self.df["signal"] = self.signal_model.predict(self.signal_model.data)

        # create prediction at reference and alternative mid points
        ref_cov = np.repeat(self.ref_exposure, self.df.shape[0])
        alt_cov = self.df.ref_mid.to_numpy()
        data = MRData(covs={**{name: ref_cov for name in self.ref_cov_names},
                            **{name: alt_cov for name in self.alt_cov_names}})
        self.df["ref_pred"] = self.signal_model.predict(data)
        self.df["alt_pred"] = self.df.ref_pred + self.df.obs

        # create prediction at fine grid
        self.pred = pred - self.ref_value

    def detect_pub_bias(self):
        # compute total obs_se
        self.df["inflated_obs_se"] = np.sqrt(
            self.df.obs_se**2 +
            self.df.signal**2*self.final_model.gamma_soln[0]
        )

        # compute residual
        self.df["residual"] = self.df.obs - self.df.signal
        self.df["weighted_residual"] = self.df.residual/self.df.inflated_obs_se

        # egger regression
        self.se_model["mean"], self.se_model["se"], self.se_model["pval"] = \
            self.egger_regression(self.df.residual[self.df.outlier == 0])
        self.se_model_all["mean"], self.se_model_all["se"], self.se_model_all["pval"] = \
            self.egger_regression(self.df.residual)

    def adjust_pub_bias(self):
        # get residual
        df = self.df[self.df.outlier == 0].reset_index(drop=True)
        residual = df.residual.to_numpy()

        # compute rank of the absolute residual
        rank = np.zeros(residual.size, dtype=int)
        rank[np.argsort(np.abs(residual))] = np.arange(1, residual.size + 1)

        # get the furthest residual according to the sign of egger regression
        sort_index = np.argsort(residual)
        if self.se_model["mean"] > 0.0:
            sort_index = sort_index[::-1]

        # compute the number of data points need to be filled
        self.num_fill = residual.size - rank[sort_index[-1]]

        # get data to fill
        df_fill = df.iloc[sort_index[:self.num_fill], :]
        df_fill.study_id = "fill_" + df_fill.study_id
        df_fill.residual = -df_fill.residual
        df_fill.obs = df_fill.signal + df_fill.residual
        df_fill.alt_pred = df_fill.obs + df_fill.ref_pred

        # combine with filled dataframe
        self.df = pd.concat([self.df, df_fill])

        # refit linear model
        self.linear_model_fill = self.fit_linear_model(self.df[self.df.outlier == 0])

        # compute score
        self.score_fill, self.inner_draws_fill, self.outer_draws_fill = self.get_score(self.linear_model_fill)

    def fit_linear_model(self, df: pd.DataFrame) -> MRBRT:
        linear_data = MRData()
        linear_data.load_df(
            df,
            col_obs="obs",
            col_obs_se="obs_se",
            col_covs=list(self.final_model.data.covs.keys()),
            col_study_id="study_id",
            col_data_id="data_id"
        )

        # create covariate model for linear model
        linear_cov_models = self.final_model.cov_models

        # create linear model
        linear_model = MRBRT(linear_data, cov_models=linear_cov_models)

        # fit linear model
        linear_model.fit_model()

        return linear_model

    @staticmethod
    def egger_regression(residual: np.ndarray) -> Tuple[float, float, float]:
        mean = np.mean(residual)
        sd = max(1, np.std(residual))/np.sqrt(residual.size)
        p = norm.cdf(0.0, loc=mean, scale=sd)
        pval = 2.0*min(p, 1 - p)
        return mean, sd, pval

    @property
    def has_pub_bias(self) -> bool:
        return self.se_model["pval"] < 0.05

    def get_score(self, model: MRBRT) -> Tuple[float, np.ndarray, np.ndarray]:
        # compute the posterior standard error of the fixed effect
        inner_fe_sd = np.sqrt(get_beta_sd(model)**2 + model.gamma_soln[0])
        outer_fe_sd = np.sqrt(get_beta_sd(model)**2 +
                              model.gamma_soln[0] + 2.0*get_gamma_sd(model))

        # compute the lower and upper signal multiplier
        inner_betas = (norm.ppf(0.05, loc=1.0, scale=inner_fe_sd),
                       norm.ppf(0.95, loc=1.0, scale=inner_fe_sd))
        outer_betas = (norm.ppf(0.05, loc=1.0, scale=outer_fe_sd),
                       norm.ppf(0.95, loc=1.0, scale=outer_fe_sd))

        # compute the lower and upper draws
        inner_draws = np.vstack([inner_betas[0]*(self.pred - self.ref_value),
                                 inner_betas[1]*(self.pred - self.ref_value)])
        outer_draws = np.vstack([outer_betas[0]*(self.pred - self.ref_value),
                                 outer_betas[1]*(self.pred - self.ref_value)])
        for i in range(2):
            inner_draws[i] -= np.interp(self.ref_exposure, self.pred_exposures, inner_draws[i])
            outer_draws[i] -= np.interp(self.ref_exposure, self.pred_exposures, outer_draws[i])

        # compute and score
        index = ((self.pred_exposures >= self.exposure_limits[0]) &
                 (self.pred_exposures <= self.exposure_limits[1]))
        sign = np.sign(self.pred[index].mean())
        score = np.min(outer_draws[:, index].mean(axis=1)*sign)

        return score, inner_draws, outer_draws

    def plot_residual(self, ax: plt.Axes) -> plt.Axes:
        # compute the residual and observation standard deviation
        residual = self.df["residual"]
        obs_se = self.df["inflated_obs_se"]
        max_obs_se = np.quantile(obs_se, 0.99)
        fill_index = self.df.study_id.str.contains("fill")

        # create funnel plot
        ax = plt.subplots()[1] if ax is None else ax
        ax.set_ylim(max_obs_se, 0.0)
        ax.scatter(residual, obs_se, color="gray", alpha=0.4)
        if fill_index.sum() > 0:
            ax.scatter(residual[fill_index], obs_se[fill_index], color="#008080", alpha=0.7)
        ax.scatter(residual[self.df.outlier == 1], obs_se[self.df.outlier == 1],
                   color='red', marker='x', alpha=0.4)
        ax.fill_betweenx([0.0, max_obs_se],
                         [0.0, -1.96*max_obs_se],
                         [0.0, 1.96*max_obs_se],
                         color='#B0E0E6', alpha=0.4)
        ax.plot([0, -1.96*max_obs_se], [0.0, max_obs_se], linewidth=1, color='#87CEFA')
        ax.plot([0.0, 1.96*max_obs_se], [0.0, max_obs_se], linewidth=1, color='#87CEFA')
        ax.axvline(0.0, color='k', linewidth=1, linestyle='--')
        ax.set_xlabel("residual")
        ax.set_ylabel("ln_rr_se")
        ax.set_title(
            f"{self.name}: egger_mean={self.se_model['mean']: .3f}, "
            f"egger_sd={self.se_model['sd']: .3f}, "
            f"egger_pval={self.se_model['pval']: .3f}",
            loc="left")
        return ax

    def plot_model(self, ax: plt.Axes) -> plt.Axes:
        # plot data
        ax.scatter(self.df.alt_mid,
                   self.df.alt_pred,
                   s=5.0/self.df.obs_se,
                   color="gray", alpha=0.5)
        index = self.df.outlier == 1
        ax.scatter(self.df.alt_mid[index],
                   self.df.alt_pred[index],
                   s=5.0/self.df.obs_se[index],
                   marker="x", color="red", alpha=0.5)

        # plot prediction
        ax.plot(self.pred_exposures, self.pred, color="#008080", linewidth=1)

        # plot uncertainties
        ax.fill_between(self.pred_exposures,
                        self.inner_draws[0],
                        self.inner_draws[1], color="#69b3a2", alpha=0.2)
        ax.fill_between(self.pred_exposures,
                        self.outer_draws[0],
                        self.outer_draws[1], color="#69b3a2", alpha=0.2)

        # plot filled model
        if self.num_fill > 0:
            ax.plot(self.pred_exposures, self.outer_draws_fill[0],
                    linestyle="--", color="gray", alpha=0.5)
            ax.plot(self.pred_exposures, self.outer_draws_fill[1],
                    linestyle="--", color="gray", alpha=0.5)
            index = self.df.study_id.str.contains("fill")
            ax.scatter(self.df.alt_mid[index],
                       self.df.alt_pred[index],
                       s=5.0/self.df.obs_se[index],
                       marker="o", color="#008080", alpha=0.5)

        # plot bounds
        for b in self.exposure_bounds:
            ax.axvline(b, linestyle="--", linewidth=1, color="k")

        # plot 0 line
        ax.axhline(0.0, linestyle="-", linewidth=1, color="k")

        # title
        title = f"{self.name}: score={self.score: .3f}"
        if self.num_fill > 0:
            title = title + f", score_fill={self.score_fill: .3f}"
        ax.set_title(title, loc="left")

        return ax


def get_gamma_sd(model: MRBRT) -> float:
    gamma = model.gamma_soln
    gamma_fisher = model.lt.get_gamma_fisher(gamma)
    return 1.0/np.sqrt(gamma_fisher[0, 0])


def get_beta_sd(model: MRBRT) -> float:
    model_specs = extract_simple_lme_specs(model)
    beta_hessian = extract_simple_lme_hessian(model_specs)
    return 1.0/np.sqrt(beta_hessian[0, 0])
