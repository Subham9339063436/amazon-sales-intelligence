"""Tests for evaluation metrics and the naive baseline."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import NaiveLastValueModel, last_observed_value
from src.metrics import regression_metrics
from tests.conftest import make_panel


class TestRegressionMetrics:
    def test_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)
        assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
        # MAPE = mean(|err/actual|) = (0.2 + 0.1 + 0.1) / 3 = 13.333...%
        assert m["mape_percent"] == pytest.approx(40 / 3)

    def test_mape_is_null_when_non_positive_targets(self):
        y_true = np.array([10.0, 0.0, 30.0])
        y_pred = np.array([10.0, 5.0, 30.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mape_percent"] is None
        # Other metrics still computed.
        assert m["mae"] == pytest.approx(5 / 3)

    def test_perfect_prediction(self):
        y = np.array([5.0, 6.0, 7.0])
        m = regression_metrics(y, y.copy())
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == pytest.approx(1.0)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            regression_metrics(np.array([]), np.array([]))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


class TestNaiveBaseline:
    def test_predicts_last_observed_value_per_product(self):
        frame = make_panel({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
        train = frame[frame["month"] <= pd.Timestamp("2019-02-01")]
        test = frame[frame["month"] > pd.Timestamp("2019-02-01")]

        model = NaiveLastValueModel().fit(train)
        preds = model.predict(test)

        # Predicts February's values for March rows.
        np.testing.assert_allclose(preds, [2.0, 20.0])
        assert model.n_fallback_ == 0

    def test_unseen_product_falls_back_to_train_mean(self):
        frame = make_panel({"A": [1.0, 2.0, 3.0]})
        train = frame.copy()
        test = pd.DataFrame(
            {
                "product": ["ZZZ"],
                "month": [pd.Timestamp("2019-06-01")],
                "sales_amount": [999.0],
            }
        )
        model = NaiveLastValueModel().fit(train)
        preds = model.predict(test)
        assert model.n_fallback_ == 1
        assert preds[0] == pytest.approx(train["sales_amount"].mean())

    def test_predict_before_fit_raises(self):
        frame = make_panel({"A": [1.0, 2.0]})
        model = NaiveLastValueModel()
        with pytest.raises(RuntimeError):
            model.predict(frame)

    def test_last_observed_value_revenue(self):
        rev = pd.DataFrame(
            {
                "ds": pd.date_range("2019-01-01", periods=3, freq="MS"),
                "y": [100.0, 200.0, 300.0],
            }
        )
        assert last_observed_value(rev) == 300.0
        # Order of rows must not matter.
        assert last_observed_value(rev.sample(frac=1, random_state=0)) == 300.0
