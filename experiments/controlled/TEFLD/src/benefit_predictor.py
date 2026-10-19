"""Small frozen regression head, saved as strict JSON rather than a pickle."""

from __future__ import annotations

import math

from .helper import write_json_file, read_json_file


def feature_subset(features, mode):
    if mode not in {"gradient", "jev", "hybrid"}:
        raise ValueError("Unknown predictor feature mode.")
    if mode in {"jev", "hybrid"} and not any(k.startswith("jev_") for k in features):
        raise ValueError("Jev observations are required for this predictor mode.")
    if mode == "gradient":
        return {k: v for k, v in features.items() if not k.startswith("jev_")}
    if mode == "jev":
        return {
            k: v
            for k, v in features.items()
            if k.startswith(("jev_", "family_"))
            or k in {"candidate_nll", "target_tokens", "sequence_tokens", "progress"}
        }
    return features


class BenefitPredictor:
    def __init__(self, payload):
        self.payload = payload

    @classmethod
    def fit(cls, rows, alpha=10.0, feature_mode="gradient"):
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        if len(rows) < 16:
            raise ValueError(
                "At least 16 fit observations required; use a development gate before deployment."
            )
        rows = [
            {**r, "features": feature_subset(r["features"], feature_mode)} for r in rows
        ]
        names = sorted(rows[0]["features"])
        if any(sorted(r["features"]) != names for r in rows):
            raise ValueError("Feature schema changed.")
        matrix = np.array(
            [[r["features"][n] for n in names] for r in rows], dtype=float
        )
        targets = np.array([r["gain"] for r in rows], dtype=float)
        if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
            raise ValueError("Nonfinite training data.")
        scaler = StandardScaler().fit(matrix)
        head = Ridge(alpha=alpha).fit(scaler.transform(matrix), targets)
        return cls(
            {
                "version": "benefit-ridge-v1",
                "feature_mode": feature_mode,
                "features": names,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coefficients": head.coef_.tolist(),
                "intercept": float(head.intercept_),
                "alpha": alpha,
                "fit_rows": len(rows),
                "probability_status": "unavailable_without_separate_calibration",
            }
        )

    def predict(self, features):
        p = self.payload
        features = feature_subset(features, p.get("feature_mode", "gradient"))
        if set(features) != set(p["features"]):
            raise ValueError("Predictor feature schema mismatch.")
        value = p["intercept"] + sum(
            c * (features[n] - mean) / scale
            for n, c, mean, scale in zip(
                p["features"], p["coefficients"], p["mean"], p["scale"]
            )
        )
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite gain prediction.")
        probability = None
        calibration = p.get("calibration")
        if calibration:
            z = calibration["intercept"] + calibration["coefficient"] * value
            probability = (
                1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            )
        return {"expected_gain_nll": value, "p_improve_calibrated": probability}

    def calibrate(self, rows, delta_min):
        from sklearn.linear_model import LogisticRegression

        labels = [int(r["gain"] > delta_min) for r in rows]
        if min(labels.count(0), labels.count(1)) < 10:
            self.payload["probability_status"] = (
                "unavailable_insufficient_outcome_classes"
            )
            return
        scores = [[self.predict(r["features"])["expected_gain_nll"]] for r in rows]
        head = LogisticRegression(C=1.0, random_state=0).fit(scores, labels)
        self.payload["calibration"] = {
            "coefficient": float(head.coef_[0, 0]),
            "intercept": float(head.intercept_[0]),
            "delta_min": float(delta_min),
            "rows": len(rows),
        }
        self.payload["probability_status"] = "calibration_fitted_requires_locked_audit"

    def save(self, path):
        write_json_file(path, self.payload)

    @classmethod
    def load(cls, path):
        return cls(read_json_file(path, {}))


def regression_metrics(rows, predictor):
    if not rows:
        raise ValueError("Empty predictor audit.")
    errors = [
        predictor.predict(r["features"])["expected_gain_nll"] - r["gain"] for r in rows
    ]
    return {
        "mae": sum(abs(x) for x in errors) / len(errors),
        "rmse": math.sqrt(sum(x * x for x in errors) / len(errors)),
        "observations": len(rows),
    }
