# import os
# import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from data_manage.K_fold.K_Fold import K_Fold
from src.data_manage.VIF import calculate_vif
from src.data_manage.csv_to_excel import csv_to_excel
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

# from src.analysis.SHAP import shap_analysis
# from sklearn.ensemble import AdaBoostRegressor
# from src.feature_engineering import average_daily
# from Optimiser.objective_function import objective_adaboost
# from data_manage.Z_Score import remove_outliers_zscore
from src.model.get_X_y import get_X_y

# from src.analysis.LIME import lime_sensitivity_analysis
# from src.Optimiser.HOA.hoa_optimizer import hoa_optimizer


# csv_to_excel(
#     "task/Dataset-8MO-Yahyavi (HVAC Efficiency).csv",
#     "data/data.xlsx",
#     sheet_name="data",
# )

DATA_PATH = "data/data.xlsx"
TARGET = "HVAC Efficiency"
df = pd.read_excel(DATA_PATH, sheet_name="data")


# vif_results
vif_results = calculate_vif(df, target=TARGET, save_to_excel=True)


X, y = get_X_y(df, target_col=TARGET)


# Count the frequency of each unique label in y


# Apply K-Fold cross-validation
model = XGBClassifier()
(
    X_train_K_fold,
    X_test_K_fold,
    y_train_K_fold,
    y_test_K_fold,
    K_Fold_Cross_Validation_Scores,
    combined_df,
) = K_Fold(X, y, n_splits=5, DATA_PATH=DATA_PATH, save_to_excel=True, model=model)

combined_df = pd.read_excel(DATA_PATH, sheet_name="DATA after K-Fold")


# ===== Balance the training dataset using Edited Nearest Neighbours (ENN) =====
print("Original training shape:", X_train_K_fold, y_train_K_fold)
enn = EditedNearestNeighbours()
X_train_K_fold, y_train_K_fold = enn.fit_resample(X_train_K_fold, y_train_K_fold)
X_test_K_fold, y_test_K_fold = enn.fit_resample(X_test_K_fold, y_test_K_fold)

print("balacne training shape:", X.shape, y.shape)

model.fit(X_train_K_fold, y_train_K_fold)


# # Helper function to average metrics
# def summarize_metrics(metrics_list):
#     return {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}


# # avg_metrics_k_fold = summarize_metrics(K_Fold_Cross_Validation_Scores)
# # print("Average Metrics:")
# # for key, value in avg_metrics_k_fold.items():
# #     print(f"{key}: {value:.4f}")


# # singleModel_result = train_model(X_train_best, y_train_best, X_test_best, y_test_best)


# # best_pos, best_RMSE, convergence = hoa_optimizer(
# #     objective_adaboost,  # our AdaBoost objective
# #     [500, 0.01],  # lower bounds: n_estimators, learning_rate
# #     [1000, 1.0],  # upper bounds
# #     2,  # dim
# #     5,  # n_agents
# #     5,  # max_iter
# #     X_train_best,
# #     y_train_best,
# #     X_test_best,
# #     y_test_best,
# # )

# # HOA_model_result = train_model(
# #     X_train_best, y_train_best, X_test_best, y_test_best, best_pos
# # )

# model = AdaBoostRegressor()
# model.fit(X_train_best, y_train_best)

# # # SHAP on the HOA model
# # sensitivity_df_shap, shap_values = shap_analysis(
# #     model=model,
# #     X_train=X_train_best,
# #     y_train=y_train_best,
# #     X_test=X_test_best,
# #     y_test=y_test_best,
# #     save_path=DATA_PATH,  # save to same Excel file
# #     sheet_name="SHAP_Sensitivity",
# # )


# sensitivity_LIME = pd.DataFrame(
#     [
#         lime_sensitivity_analysis(
#             model=model,
#             X_train=X_train_best,
#             y_train=y_train_best,
#             X_test=X_test_best,
#             y_test=y_test_best,
#             sample_index=5,
#             epsilon=0.05,
#         )
#     ]
# )


# # print("\nBest AdaBoost Params:", best_pos)
# # print("Best RMSE:", best_RMSE)
