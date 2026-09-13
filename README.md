# COVID-19 Risk Prediction Model

A multivariate logistic regression model that estimates an individual's
probability of contracting COVID-19 from three predictors: number of
symptoms, distance from a known positive case, and face mask usage.

Completed June 2024 as a mathematics and computer science research essay.
Published here as an archive of the original work.

## Research question

To what extent can a multivariate logistic regression model predict the
probability of an individual contracting COVID-19?

## Data

Survey data from 40 individuals, collected by random sampling:
25 positive cases and 15 negative cases.

| Feature | Encoding |
|---|---|
| Number of symptoms | 1–5 (dry cough, loss of taste, loss of smell, fatigue, fever — CDC list) |
| Distance from positive case | 1, 2, 6, or 10 meters |
| Face mask | 1 = wearing, 0 = not wearing |
| Outcome | 1 = positive, 0 = negative |

## Approach

- Multivariate logistic regression via scikit-learn's `LogisticRegression`
- 80/20 train/test split (`train_test_split`, `random_state=5`)
- Classification threshold at 0.5
- Evaluated with a ROC curve and AUC, compared against a no-skill baseline

The essay derives the logistic equation from first principles — odds, log
odds, the sigmoid function, and maximum likelihood estimation — before
delegating coefficient estimation to scikit-learn.

## Results

Fitted coefficients:
- intercept = 1.9281
- symptoms = 0.5608
- distance = -0.3345
- face mask = -1.3821


Signs behave as expected: more symptoms raises predicted risk, greater
distance lowers it, and wearing a mask lowers it substantially.

Predicted probabilities for a person with 3 symptoms:

| Scenario | Predicted risk |
|---|---|
| 2 m away, no mask | 94.99% |
| 2 m away, masked | 82.63% |
| 8 m away, no mask | 71.80% |
| 8 m away, masked | 38.99% |

Mask usage reduced predicted risk by 12.4 percentage points at close range
and 32.8 points at distance.

## Limitations

The sample is 40 people, which leaves only 8 records in the test set — far
too few for the AUC to be a reliable estimate of generalization. The model
is a demonstration of the method, not a validated clinical tool. Duration
of exposure and indoor vs. outdoor setting are both absent and would likely
matter more than distance alone.

## Repository contents

- `essay.pdf` — full derivation, methodology, and results
- `model.py` — model fitting and ROC evaluation *(see note below)*

## Running it

```bash
pip install -r requirements.txt
python model.py
```

Note: this code was written in 2024 and has not been re-run against current
library versions.
