"""
gesture_system/trainer.py
==========================
GestureTrainer — dataset collection and ML model training pipeline.

Two workflows
─────────────
1. COLLECT mode
   Reads webcam, runs MediaPipe, saves feature vectors labelled by gesture.
   Press the keyboard shortcut for the target gesture, then hold the pose.
   Features are appended to dataset/<gesture>.npy.

2. TRAIN mode
   Loads all .npy feature files from the dataset directory, trains a
   RandomForestClassifier (or user-supplied estimator), validates with
   stratified k-fold, and saves the model to models/gesture_model.pkl.

3. EVALUATE mode
   Loads the model and runs a live confusion-matrix evaluation against
   held-out test data.

Usage
─────
    # Collect training samples
    python -m gesture_system.trainer --mode collect --gesture open_hand

    # Train from collected data
    python -m gesture_system.trainer --mode train

    # Quick live test
    python -m gesture_system.trainer --mode evaluate
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("jarvis.gesture.trainer")

DATASET_DIR = Path("gesture-dataset")
MODEL_DIR   = Path("models")
MODEL_PATH  = MODEL_DIR / "gesture_model.pkl"

SAMPLES_PER_GESTURE = 300   # target samples per class for collection


def collect(
    gesture_name: str,
    n_samples: int = SAMPLES_PER_GESTURE,
    camera_id:  int = 0,
) -> None:
    """
    Interactively collect feature samples for one gesture class.

    Saves to gesture-dataset/<gesture_name>.npy (appends if file exists).
    """
    from gesture_system.camera import CameraManager
    from gesture_system.hand_tracking import HandTracker
    from gesture_system.feature_extraction import FeatureExtractor

    out_path = DATASET_DIR / f"{gesture_name}.npy"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    existing: np.ndarray | None = None
    if out_path.exists():
        existing = np.load(out_path)
        log.info("Appending to existing dataset: %d samples", len(existing))

    cam = CameraManager(device=camera_id)
    tracker = HandTracker(max_hands=1, draw=True)
    fe = FeatureExtractor()

    cam.start()
    tracker.start()

    features_list: list[np.ndarray] = []
    collecting = False
    collected  = 0

    print(f"\nCollect gesture: {gesture_name.upper()}")
    print("  Press SPACE to start/stop collecting")
    print(f"  Target: {n_samples} samples")
    print("  Press Q to quit\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, hands = tracker.process(frame)

        if collecting and hands:
            vec = fe.extract(hands[0])
            features_list.append(vec)
            collected += 1

        # HUD
        status = f"COLLECTING ({collected}/{n_samples})" if collecting else "READY — press SPACE"
        color  = (0, 255, 100) if collecting else (0, 229, 255)
        cv2.putText(annotated, f"Gesture: {gesture_name}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2)
        cv2.putText(annotated, status, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Gesture Collector — Jarvis 2.0", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
        elif key == ord('q') or collected >= n_samples:
            break

    cam.stop()
    tracker.stop()
    cv2.destroyAllWindows()

    if features_list:
        new_data = np.stack(features_list)
        if existing is not None:
            combined = np.vstack([existing, new_data])
        else:
            combined = new_data
        np.save(out_path, combined)
        log.info("Saved %d samples → %s  (total: %d)", len(features_list), out_path, len(combined))
    else:
        log.warning("No samples collected for %s", gesture_name)


def train(
    dataset_dir: Path = DATASET_DIR,
    model_path:  Path = MODEL_PATH,
    n_estimators: int = 200,
    test_size:   float = 0.2,
) -> None:
    """
    Train a RandomForestClassifier from the collected dataset.
    Saves the model + class list to model_path.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import sklearn

    log.info("Loading dataset from %s …", dataset_dir)

    X_list, y_list = [], []
    for npy in sorted(dataset_dir.glob("*.npy")):
        gesture = npy.stem
        data    = np.load(npy)
        X_list.append(data)
        y_list.extend([gesture] * len(data))
        log.info("  %s: %d samples", gesture, len(data))

    if not X_list:
        log.error("No .npy files found in %s — run --mode collect first", dataset_dir)
        return

    X = np.vstack(X_list)
    y = np.array(y_list)
    classes = sorted(set(y_list))
    log.info("Total: %d samples  %d classes: %s", len(X), len(classes), classes)

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    # Cross-validation
    log.info("Running 5-fold cross-validation …")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    log.info("CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Final fit on all data
    model.fit(X, y)
    log.info("Final training accuracy: %.3f", model.score(X, y))

    # Feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    from gesture_system.feature_extraction import FeatureExtractor
    feat_names = FeatureExtractor.feature_names()
    log.info("Top 10 features:")
    for i in top_idx:
        log.info("  %-30s  %.4f", feat_names[i], importances[i])

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "classes": list(model.classes_), "sklearn_version": sklearn.__version__}
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    log.info("Model saved → %s", model_path)

    # Classification report
    pred = model.predict(X)
    print("\n" + classification_report(y, pred, target_names=classes))


def evaluate(camera_id: int = 0) -> None:
    """Live evaluation: show real-time predictions with class probabilities."""
    from gesture_system.camera import CameraManager
    from gesture_system.hand_tracking import HandTracker
    from gesture_system.feature_extraction import FeatureExtractor
    from gesture_system.gesture_classifier import GestureClassifier

    clf = GestureClassifier(model_path=MODEL_PATH)
    if not clf.load_model():
        log.error("No model found at %s — train first", MODEL_PATH)
        return

    cam     = CameraManager(device=camera_id)
    tracker = HandTracker(max_hands=1, draw=True)
    fe      = FeatureExtractor()
    cam.start()
    tracker.start()

    print("\nLive evaluation — press Q to quit\n")
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, hands = tracker.process(frame)
        if hands:
            result = clf.classify(hands)
            label  = f"{result.gesture}  {result.confidence:.0%}  [{result.mode}]"
            cv2.putText(annotated, label, (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 255), 2)
            # Top-3 scores
            if result.raw_scores:
                top3 = sorted(result.raw_scores.items(), key=lambda x: -x[1])[:3]
                for i, (g, s) in enumerate(top3):
                    cv2.putText(annotated, f"  {g}: {s:.2f}",
                                (15, 70 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Gesture Evaluation — Jarvis 2.0", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    tracker.stop()
    cv2.destroyAllWindows()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Gesture dataset collection + training")
    parser.add_argument("--mode",    choices=["collect", "train", "evaluate"], required=True)
    parser.add_argument("--gesture", default="open_hand", help="Gesture name for collect mode")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_GESTURE)
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--estimators", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.gesture, n_samples=args.samples, camera_id=args.camera)
    elif args.mode == "train":
        train(n_estimators=args.estimators)
    elif args.mode == "evaluate":
        evaluate(camera_id=args.camera)
