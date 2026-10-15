# Bundled datasets

All five public datasets ship with PyAutoCausal and load as pandas DataFrames. No download is needed. Copy a frame before preparing it so you do not modify the module-level example data:

```python
from pyautocausal.datasets import minimum_wage

data = minimum_wage.copy()
assert list(data.columns) == ["id_unit", "t", "treat", "y", "lpop"]
```

## Public exports in 0.1.2

| Import name | Rows × columns | Contents and preparation |
| --- | --- | --- |
| `minimum_wage` | 2,500 × 5 | Prepared panel: 500 units over 2003–2007; `treat` is zero before adoption and one afterward; `lpop` is a covariate. |
| `california_prop99` | 1,209 × 8 | 39 states over 1970–2000. Rename/select the outcome and design columns as in the [California example](../../examples/scripts/california_prop99.py). |
| `lalonde` | 614 × 10 | Cross-sectional data with `treat`, `y`, `id_unit` and candidate controls. Select and encode covariates explicitly; `race` is categorical. |
| `mpdta` | 2,500 × 6 | Raw minimum-wage data. Its `treat` indicates ever-treated units, so it is **not ready for the panel factory**. |
| `PENN` | 5,328 × 5 | Columns `country`, `year`, `log_gdp`, `dem` and `educ`. The loader handles the underlying semicolon delimiter. Choose an appropriate design and rename columns before estimation. |

These names are defined in [`datasets.__all__`](__init__.py). Other files in this directory, including `lalonde_raw.csv`, are not named public exports. A bundled dataset is an example input, not a guarantee that every estimator supports its design or that default adjustment choices identify an effect.

## Minimum-wage preparation

The minimum-wage example data is associated with Callaway and Sant’Anna (2021). Use `minimum_wage` for the prepared county-level teen-employment panel. Merely renaming the raw `mpdta` columns would leave treatment active before adoption. The transformation below reproduces the prepared data:

```python
from pandas.testing import assert_frame_equal
from pyautocausal.datasets import mpdta, minimum_wage

data = mpdta.rename(
    columns={"countyreal": "id_unit", "year": "t", "lemp": "y"}
).copy()
data["treat"] = (
    (data["first.treat"] > 0) & (data["t"] >= data["first.treat"])
).astype(int)
data = data[["id_unit", "t", "treat", "y", "lpop"]]
assert_frame_equal(data, minimum_wage)
```

See the [minimum-wage script](../../examples/scripts/minimum_wage.py) for fitting and exporting this panel.

## California Proposition 99

The California data is from Abadie, Diamond and Hainmueller (2010). The bundled example uses cigarette sales (`cigsale`) as the outcome, `state` as the unit, `year` as time and observed `treated` status:

```python
from pyautocausal.datasets import california_prop99

data = california_prop99.rename(
    columns={"state": "id_unit", "year": "t", "treated": "treat", "cigsale": "y"}
)[["id_unit", "t", "treat", "y"]].copy()
```

The example intentionally omits covariates with missing values rather than filling them with an arbitrary sentinel. Additional adjustment needs a justified specification and explicit missing-data handling. The [data requirements](../../docs/data-requirements.md) and [methods guide](../../docs/causal-methods.md) describe those contracts.

## External datasets

Criteo uplift data is not bundled and is not a public dataset export. Remote readers and download dependencies are not supplied by a PyAutoCausal extra. Acquire and prepare external data separately, then validate its outcome, treatment, covariates and design against the selected estimator.
