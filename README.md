# Lab-assignment-1
# California Housing Regression: Linear vs. Ridge

A compact scikit-learn lab exercise that uses the California Housing dataset to compare ordinary linear regression with Ridge regression. The notebook samples a reproducible subset of the data, standardizes the input features, evaluates models with five-fold cross-validation using the coefficient of determination (R²), and plots the results.

## Project Overview

The notebook is organized into three tasks:

1. **Dataset setup** — load the California Housing dataset, select 1,000 records with a fixed random seed, and separate the target from the input features.
2. **Model training and evaluation** — calculate mean five-fold cross-validation R² scores for a linear regression baseline and Ridge models with different regularization strengths.
3. **Visualization** — display a bar chart comparing the models' mean cross-validation scores.

This is an educational experiment. It is intended to demonstrate a basic regression workflow and the effect of Ridge regularization; it is not a production-ready housing valuation system.

## Dataset

The notebook uses [`fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn. On the first run, scikit-learn may need to download the dataset.

The input features are:

| Feature | Description |
| --- | --- |
| `MedInc` | Median income in the census block group |
| `HouseAge` | Median house age in the census block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Census block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Latitude of the census block group |
| `Longitude` | Longitude of the census block group |

The target column is `MedHouseVal`, the median house value for the block group, expressed in units of **$100,000** in the source dataset. For example, a target value of `2.0` corresponds to $200,000 in the dataset's units.

The experiment uses a sample of 1,000 rows selected with `random_state=42`, so the same sample is selected each run when the underlying dataset version is unchanged.

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook or Google Colab

Install the Python packages with:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

## Getting Started

### Run in Google Colab

1. Upload `Untitled0.ipynb` to Google Colab, or open it from Google Drive.
2. Run the notebook cell from top to bottom.
3. Allow the runtime to download the California Housing dataset if prompted by the environment.
4. Review the printed sample rows, cross-validation scores, and comparison chart.

### Run locally

1. Save the notebook in your working directory.
2. Install the requirements above.
3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `Untitled0.ipynb` in the browser and run all cells in order.

The provided notebook currently places the full workflow in one code cell. The code can also be split into separate notebook cells by task for easier exploration.

## Workflow Details

### 1. Prepare the data

The notebook calls `fetch_california_housing(as_frame=True)` and samples 1,000 rows from the resulting DataFrame. It then creates:

- `X`: the eight predictor columns, with `MedHouseVal` removed.
- `y`: the `MedHouseVal` target column.

A `StandardScaler` transforms the features so each feature has approximately zero mean and unit variance. The standardized feature matrix is divided into training and test sets with an 80/20 split and `random_state=42`.

### 2. Evaluate regression models

The training partition is evaluated with five-fold cross-validation (`cv=5`) and the R² scoring metric:

- **Linear Regression** — ordinary least squares, used as the baseline.
- **Ridge Regression** — linear regression with an L2 penalty, evaluated at `alpha=0.1`, `10.0`, and `1000.0`.

The notebook reports the mean of the five validation-fold R² scores for each model. A higher R² indicates that the model explains more of the variation in the target on the validation folds. An R² of 1 is a perfect fit; 0 is roughly equivalent to predicting the mean target for each validation example. R² can be negative when a model performs worse than that baseline.

### 3. Compare results visually

The final chart shows the mean cross-validation R² for the baseline and each Ridge alpha. Bar labels display the scores to three decimal places. Increasing `alpha` strengthens the penalty on large coefficients; the experiment illustrates how regularization strength can affect predictive performance.

## Results Recorded in the Notebook

The saved notebook output reports the following mean five-fold R² scores:

| Model | Mean 5-fold R² |
| --- | ---: |
| Linear Regression | 0.6457 |
| Ridge (`alpha=0.1`) | 0.6457 |
| Ridge (`alpha=10.0`) | 0.6443 |
| Ridge (`alpha=1000.0`) | 0.3574 |

These are the outputs stored in the supplied notebook. Scores may change if the dataset, sampling, preprocessing, or evaluation procedure changes.

## Important Implementation Notes

- The notebook creates an 80/20 train/test split, but the test set is not used in the current evaluation or reporting. The displayed scores come from cross-validation on the training partition, not from a final test-set evaluation.
- In the current code, `StandardScaler` is fit on all 1,000 sampled rows before the train/test split and cross-validation. For a stricter evaluation, put scaling and regression inside a scikit-learn `Pipeline` and pass that pipeline to cross-validation. This ensures the scaler is fit only on each training fold and avoids information from validation folds influencing preprocessing.
- The notebook does not tune `alpha` beyond the three explicitly listed values. A larger or more systematic search could use `GridSearchCV` with a pipeline.
- The experiment uses a subset of the full dataset to keep the exercise lightweight. Its scores should not be treated as a definitive benchmark for the complete dataset.

## Customization

You can modify the notebook to explore the workflow further:

- Change the sample size or `random_state`.
- Add or remove values in `alphas` to study the effect of regularization.
- Add models such as `Lasso` or `ElasticNet` for comparison.
- Use a `Pipeline` with `StandardScaler` and evaluate it with cross-validation.
- Fit the selected model on the full training partition and report performance on the held-out test set.
- Add metrics such as mean absolute error (MAE) or root mean squared error (RMSE).

## Repository Contents

```text
.
└── Untitled0.ipynb    # Notebook containing the data preparation, evaluation, and plot
```

## License and Data Attribution

The notebook itself does not specify a project license. Add a license file before redistributing or reusing the code if a particular license is required. The dataset is provided through scikit-learn; refer to the [California Housing dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) for dataset details and attribution information.
