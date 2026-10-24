import logging

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("adult-income")


def train_model(x_train, y_train):
    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42)
    logger.info(f"Model parameters: {model.get_params()}")
    model.fit(x_train, y_train)
    return model
