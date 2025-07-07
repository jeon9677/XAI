from .models import get_model_spatialmodelv3

def train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=64, mc=True):
    model = get_model_spatialmodelv3(mc=mc)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return model, history
