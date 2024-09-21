from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
n_features = 3

@ignore_warnings(category=ConvergenceWarning)

def lasso_features(df, tf):
    n_alphas = 100
    alphas = np.logspace(-5,2,num=n_alphas)
    for i in range(n_alphas):
        beta = Lasso(fit_intercept=True, alpha=alphas[i]).fit(df, tf).coef_.ravel()
        nonzero_mask = (np.abs(beta) >= 1e-2)
        if np.sum(nonzero_mask) <= n_features:
            break
    
    print(beta)
    print(df.columns[nonzero_mask])

def sparse_mixed_effects(df):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = StandardScaler()
    scaler.fit(poly.fit_transform(df))
    df_poly = pd.DataFrame(data=scaler.transform(poly.fit_transform(df)),
                       columns=poly.get_feature_names_out(df.columns))                  
    return df_poly

def ten_mixed_features(df, tf):
    n_alphas = 100
    alphas = np.logspace(-1,2,num=n_alphas)
    df_poly = sparse_mixed_effects(df)
    for i in range(n_alphas):
        beta = Lasso(fit_intercept=True, alpha=alphas[i]).fit(df_poly, tf).coef_.ravel()
        nonzero_mask = (np.abs(beta) >= 1e-2)
        #nonzero_mask = (beta != 0)
        if np.sum(nonzero_mask) <= 10:
            break
    
    print(beta)
    print(df_poly.columns[nonzero_mask])
        