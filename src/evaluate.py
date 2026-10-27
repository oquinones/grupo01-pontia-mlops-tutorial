import logging

from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger("adult-income")


def evaluate(model, x_test, y_test):
    logger.info("Evaluating model...")
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Classification Report:\n{report}")
    return
