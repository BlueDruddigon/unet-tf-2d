from keras.losses import CategoricalCrossentropy

from .dice import dice_loss


def dice_ce_loss(alpha: float = 0.5):
    cross_entropy_loss = CategoricalCrossentropy()
    
    def loss_fn(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        cross_entropy = cross_entropy_loss(y_true, y_pred)
        combined_loss = alpha*dice + (1.-alpha) * cross_entropy
        return combined_loss
    
    return loss_fn
