import numpy as np
from sklearn.linear_model import lars_path


def select_optimal_lambda(residuals, non_zero_count, nscans, criteria="bic"):
    """_summary_

    Parameters
    ----------
    residuals : _type_
        _description_
    non_zero_count : _type_
        _description_
    nscans : _type_
        _description_
    criteria : str, optional
        _description_, by default "bic"

    Returns
    -------
    _type_
        _description_
    """
    if criteria == "bic":
        # BIC regularization curve
        optimization_curve = nscans * np.log(residuals) + np.log(nscans) * non_zero_count
    elif criteria == "aic":
        # AIC regularization curve
        optimization_curve = nscans * np.log(residuals) + 2 * non_zero_count

    # Optimal lambda is given by the minimum of the optimization curve
    idx_optimal_lambda = np.argmin(optimization_curve)

    return idx_optimal_lambda


def solve_regularization_path(X, y, nlambdas, criteria="bic"):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    nlambdas : _type_
        _description_
    criteria : str, optional
        _description_, by default "bic"

    Returns
    -------
    _type_
        _description_
    """
    nscans = y.shape[0]

    # LARS path
    lambdas, _, coef_path = lars_path(
        X,
        np.squeeze(y),
        method="lasso",
        Gram=np.dot(X.T, X),
        Xy=np.dot(X.T, np.squeeze(y)),
        max_iter=nlambdas - 1,
        eps=1e-9,
    )

    # Compute residuals for model selection criteria (BIC and AIC)
    residuals = np.sum((np.repeat(y, nlambdas, axis=-1) - np.dot(X, coef_path)) ** 2, axis=0)

    optimal_lambda_idx = select_optimal_lambda(
        residuals, np.count_nonzero(coef_path, axis=0), nscans, criteria
    )

    return coef_path[:, optimal_lambda_idx], lambdas[optimal_lambda_idx]
