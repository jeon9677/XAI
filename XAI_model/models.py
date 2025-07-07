# my_spatial_model/models.py

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Multiply, Add

def get_dropout(x, p=0.5, mc=False):
    return Dropout(p)(x, training=mc)

def get_model_spatialmodelv3(
    n_roi=148, 
    coord_dim=3, 
    n_covariates=2, 
    n_random_effects=4, 
    hidden_dim=128,
    dropout_p=0.1,
    mc=False
):
    # 1️⃣ Spatial coordinates input
    s_input = Input(shape=(n_roi, coord_dim), name="spatial_coordinates")

    # 2️⃣ Covariates
    covariates = []
    for i in range(n_covariates):
        cov = Input(shape=(n_roi, 1), name=f"cov_{i+1}")
        covariates.append(cov)

    # 3️⃣ Spatial random effects
    s_random_list = []
    for i in range(n_random_effects):
        s_random = Input(shape=(None, 1), name=f"s_random_{i+1}")
        s_random_list.append(s_random)

    # 4️⃣ Beta blocks (예: covariates 2개라면 beta1, beta2)
    def beta_block(x):
        x = Dense(hidden_dim, activation='relu')(x)
        x = get_dropout(x, dropout_p, mc)
        x = Dense(1, activation='linear')(x)
        return x

    betas = []
    for i in range(n_covariates):
        betas.append(beta_block(s_input))

    # 5️⃣ y_s = covariate * beta 
    y_s = Add()([
        Multiply()([cov, beta]) for cov, beta in zip(covariates, betas)
    ])

    # 6️⃣ Spatial random blocks
    def random_block(x):
        x = Dense(hidden_dim, activation='relu')(x)
        x = get_dropout(x, dropout_p, mc)
        x = Dense(1, activation='linear')(x)
        return x

    spatial_randoms = [random_block(s) for s in s_random_list]

    # 7️⃣ 최종 output
    final = Add(name="final_output")([y_s] + spatial_randoms)

    # 8️⃣ Model 정의
    inputs = [s_input] + covariates + s_random_list
    model = Model(inputs=inputs, outputs=final)

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model
