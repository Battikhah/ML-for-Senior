"""Small model families trained from scratch, with a shared logit interface."""


def build_model(config: dict):
    """Keep architecture choice independent of training, tracking and export logic."""
    import tensorflow as tf
    k = tf.keras; layers = k.layers
    f = config['features']; w = config['width']; family = config['family']
    inputs = k.Input((f['frames'], f['coefficients'], 1), name='mfcc')
    x = inputs
    if family == 'mlp':
        x = layers.Flatten()(x)
        x = layers.Dense(w * 2, activation='relu')(x)
        x = layers.Dropout(config['dropout'])(x)
        x = layers.Dense(w, activation='relu')(x)
    elif family == 'temporal':
        # Collapse coefficient width once, then convolve only along time.
        x = layers.Conv2D(w, (3, f['coefficients']), activation='relu')(x)
        x = layers.Conv2D(w * 2, (3, 1), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif family in {'cnn', 'residual', 'depthwise'}:
        for index, channels in enumerate([w, w * 2, w * 4]):
            if family == 'depthwise' and index:
                x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
                x = layers.Conv2D(channels, 1, activation='relu')(x)
            elif family == 'residual':
                shortcut = layers.Conv2D(channels, 1, padding='same')(x)
                x = layers.Conv2D(channels, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(channels, 3, padding='same')(x)
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)
            else:
                x = layers.Conv2D(channels, 3, padding='same', activation='relu')(x)
            if index < 2:
                x = layers.MaxPooling2D()(x)
            x = layers.Dropout(config['dropout'])(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(f'Not a Keras family: {family}')
    return k.Model(inputs, layers.Dense(1, name='logit')(x), name=family)


def build_classical(config: dict):
    """Provide inexpensive non-neural controls before assuming a CNN is needed."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    weighted = 'balanced' if config['class_weight'] else None
    family = config['family']
    if family == 'logistic':
        return LogisticRegression(max_iter=1000, class_weight=weighted, random_state=config['seed'])
    if family == 'svm':
        return SVC(C=1., probability=True, class_weight=weighted, random_state=config['seed'])
    if family == 'forest':
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      class_weight=weighted, n_jobs=2, random_state=config['seed'])
    if family == 'histogram':
        return HistGradientBoostingClassifier(max_iter=100, class_weight=weighted,
                                              random_state=config['seed'])
    raise ValueError(f'Not a classical family: {family}')
