# Written in REPL environment

# %%

# Option 2A

import os

import pandas as pd

from config import DATASETS_DIR

DATASET1_FILENAME = "lab2_lin2.csv"
DATASET2_FILENAME = "lab2_poly1.csv"

DATASET1_PATH = os.path.join(DATASETS_DIR, '2', DATASET1_FILENAME)
DATASET2_PATH = os.path.join(DATASETS_DIR, '2', DATASET2_FILENAME)

# %%
# 1.1

df1 = pd.read_csv(DATASET1_PATH)
df1.drop(columns=df1.columns[0], inplace=True)
df1.describe()

# %%
# 1.2

from sklearn.model_selection import train_test_split

y_label = 'Price'
X_labels = list(df1.columns.drop([y_label]))

X_train, X_test, y_train, y_test = train_test_split(
    df1[X_labels],
    df1[[y_label]],
    train_size=0.8,
    random_state=42
)
print(X_train[:5])
print(X_test[:5])
print(y_train[:5])
print(y_test[:5])

# %%
# Normalize data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = \
    pd.DataFrame(scaler.fit_transform(X_train), columns=X_labels)
X_test_scaled = \
    pd.DataFrame(scaler.transform(X_test), columns=X_labels)
y_train_scaled = \
    pd.DataFrame(scaler.fit_transform(y_train), columns=[y_label])
y_test_scaled = \
    pd.DataFrame(scaler.transform(y_test), columns=[y_label])

print(X_train_scaled)
print(y_train_scaled)


# %%

import seaborn as sns


X_train_scaled_labeled = X_train_scaled.copy()
X_test_scaled_labeled = X_test_scaled.copy()
y_train_scaled_labeled = y_train_scaled.copy()
y_train_scaled_labeled['Dataset'] = 'Train'
y_test_scaled_labeled = y_test_scaled.copy()
y_test_scaled_labeled['Dataset'] = 'Test'

combined_df = pd.concat([
    pd.concat([X_train_scaled_labeled, y_train_scaled_labeled], axis=1),
    pd.concat([X_test_scaled_labeled, y_test_scaled_labeled], axis=1),
], axis=0)
melted_df = pd.melt(
    combined_df,
    id_vars=['Dataset'],
    value_vars=combined_df.columns,
    var_name='Feature',
    value_name='Value'
)


sns.violinplot(data=melted_df, x='Feature', y='Value', hue='Dataset', split=True, inner='quart')

# %%
# 1.3
# Ordinary linear model

import numpy as np

from sklearn.linear_model import LinearRegression
from functools import partial

def print_model_coefs(
    model,
    coefs_labels,
    predicted_label
):
    coefs = model.coef_.flatten()
    print(f"{predicted_label} = {model.intercept_[0]}", end = ' ')
    for i, coef_label in enumerate(coefs_labels):
        print(f"+ {coefs[i]:.2f} * {coef_label}", end = ' ')
    print()

lin_regressor = LinearRegression()
lin_regressor.fit(
    X_train_scaled,
    y_train_scaled
)

print_model_coefs_binded = partial(print_model_coefs, coefs_labels=X_labels, predicted_label=y_label)

print_model_coefs_binded(lin_regressor)

# %%
# 1.4
# Metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


def print_scores(data_map):
    scores_map = {
        "R2": r2_score,
        "MAPE": mean_absolute_percentage_error,
        "MAE": mean_absolute_error,
    }
    for dataset_label, (true_data, pred_data) in data_map.items():
        for score_label, score_func in scores_map.items():
            print(f"{score_label}={score_func(true_data, pred_data)} for {dataset_label} data")

data_map = {
    "train": (y_train_scaled, lin_regressor.predict(X_train_scaled)),
    "test": (y_test_scaled, lin_regressor.predict(X_test_scaled))
}
print_scores(data_map)

# %%
# Generic function to compare linear models

import copy


def cmp_models(ref, new, start, stop, num):
    """If new model better than reference it outputs its specs, else
    it prints nothing."""

    best_diff  = None
    best_model = None
    for _, alpha in enumerate(np.linspace(start, stop, num=num)):
        model = new(alpha=alpha)
        model.fit(
            X_train_scaled,
            y_train_scaled
        )

        diff = \
            model.score(X_test_scaled, y_test_scaled) - \
            ref.score(X_test_scaled, y_test_scaled)

        if best_diff is None or diff - best_diff >= 1e-15:
            best_diff = diff
            best_model = copy.copy(model)
            if diff >= 1e-15:
                print("New model is better than the old one")
                break

    print(f"alpha={best_model.alpha}")
    print_model_coefs_binded(best_model)

    data_map = {
        "train": (y_train_scaled, best_model.predict(X_train_scaled)),
        "test": (y_test_scaled, best_model.predict(X_test_scaled))
    }
    print_scores(data_map)



# %%
# 1.5.1
# Lasso regression

from sklearn.linear_model import Lasso

cmp_models(lin_regressor, Lasso, 0.001, 1.0, 1000)

# %%
# 1.5.2
# Ridge regression

from sklearn.linear_model import Ridge

cmp_models(lin_regressor, Ridge, 0.001, 1.0, 1000)

# %%
# 1.5.3
# ElasticNet regression

from sklearn.linear_model import ElasticNet

cmp_models(lin_regressor, ElasticNet, 0.001, 1.0, 1000)

# %%

df2 = pd.read_csv(DATASET2_PATH)
df2.drop(columns=df2.columns[0], inplace=True)
df2.describe()

# %%

from sklearn.model_selection import train_test_split

y_label = 'Efficiency'
X_labels = df2.columns.drop([y_label])

X_train, X_test, y_train, y_test = train_test_split(
    df2[X_labels],
    df2[[y_label]],
    train_size=0.8,
    random_state=42
)
print(X_train[:5])
print(X_test[:5])
print(y_train[:5])
print(y_test[:5])

# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = \
    pd.DataFrame(scaler.fit_transform(X_train), columns=X_labels)
X_test_scaled = \
    pd.DataFrame(scaler.transform(X_test), columns=X_labels)
y_train_scaled = \
    pd.DataFrame(scaler.fit_transform(y_train), columns=[y_label])
y_test_scaled = \
    pd.DataFrame(scaler.transform(y_test), columns=[y_label])

print(X_train_scaled)
print(y_train_scaled)

# %%

import matplotlib.pyplot as plt
import seaborn as sns


X_train_scaled_labeled = X_train_scaled.copy()
X_test_scaled_labeled = X_test_scaled.copy()
y_train_scaled_labeled = y_train_scaled.copy()
y_train_scaled_labeled['Dataset'] = 'Train'
y_test_scaled_labeled = y_test_scaled.copy()
y_test_scaled_labeled['Dataset'] = 'Test'

combined_df = pd.concat([
    pd.concat([X_train_scaled_labeled, y_train_scaled_labeled], axis=1),
    pd.concat([X_test_scaled_labeled, y_test_scaled_labeled], axis=1),
], axis=0)
melted_df = pd.melt(
    combined_df,
    id_vars=['Dataset'],
    value_vars=combined_df.columns,
    var_name='Feature',
    value_name='Value'
)


sns.violinplot(data=melted_df, x='Feature', y='Value', hue='Dataset', split=True, inner='quart')

# %%
# 2.3

# Train on unscaled data just for convineance
lin_regressor = LinearRegression()
lin_regressor.fit(
    X_train,
    y_train
)

print_model_coefs_binded = partial(print_model_coefs, coefs_labels=X_labels, predicted_label=y_label)

print_model_coefs_binded(lin_regressor)
print_scores(
    {
    "train": (y_train, lin_regressor.predict(X_train)),
    "test": (y_test, lin_regressor.predict(X_test))
    }
)

# %%

sns.scatterplot(df2, x=X_labels[0], y=y_label, label="Dataset")

x_line = np.linspace(df2[y_label].min(), df2[y_label].max(), 5)
sns.lineplot(
    x=np.linspace(df2[y_label].min(), df2[y_label].max(), 5),
    y=lin_regressor.intercept_[0] + lin_regressor.coef_[0][0] * x_line,
    label='Model')

# %%
# 2.4
# Polynomial regression with degree selection

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
import matplotlib.pyplot as plt

degrees = range(1, 10)

train_r2 = []
test_r2 = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train_scaled)

    train_r2.append(model.score(X_train_poly, y_train_scaled))
    test_r2.append(model.score(X_test_poly, y_test_scaled))

# %%
# Plot R2 vs degree

plt.figure(figsize=(8, 5))

plt.plot(degrees, train_r2, label='Train R2')
plt.plot(degrees, test_r2, label='Test R2')

plt.xlabel('Polynomial degree')
plt.ylabel('R2 score')
plt.title('R2 vs Polynomial Degree')
plt.legend()

plt.show()

# %%
# Choose best degree (max test R2)

best_degree = degrees[np.argmax(test_r2)]
print(f"Best degree: {best_degree}")

# %%
# 2.5

poly = PolynomialFeatures(degree=best_degree)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

best_model = LinearRegression()
best_model.fit(X_train_poly, y_train_scaled)

# %%

print_model_coefs_binded = partial(
    print_model_coefs,
    coefs_labels=poly.get_feature_names_out(X_labels),
    predicted_label=y_label)
print_model_coefs_binded(best_model)
print_scores({
    "train": (y_train_scaled, best_model.predict(X_train_poly)),
    "test": (y_test_scaled, best_model.predict(X_test_poly))
})

# %%
import numpy as np
import seaborn as sns

feature = X_labels[0]

X_scaled = pd.concat([X_train_scaled, X_test_scaled])
y_scaled = pd.concat([y_train_scaled, y_test_scaled])

df2_scaled = pd.concat([X_scaled, y_scaled], axis=1)
sns.scatterplot(data=df2_scaled, x=feature, y=y_label, label='Data')

x_line = np.linspace(df2_scaled[feature].min(), df2_scaled[feature].max(), 100)

# Enforce other features with mean
mean_vals = X_train_scaled.mean()

X_line = pd.DataFrame({
    col: mean_vals[col] for col in X_labels
}, index=range(len(x_line)))

X_line[feature] = x_line

X_line_poly = poly.transform(X_line)
y_line = best_model.predict(X_line_poly)

sns.lineplot(x=x_line, y=y_line.flatten(), label='Polynomial model')

plt.title(f"Polynomial regression (degree={best_degree})")
plt.show()
