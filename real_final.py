from tensorflow.keras.models import Model,load_model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD
import sys
from scipy.stats import norm
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate,Multiply, Add,Lambda
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,concatenate
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time


def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def get_model_spatialmodelv3(mc=False, act="relu"):
    # Spatial coordinates input
    s_input = Input(shape=(148,3), name='spatial_coordinates')  # 3D spatial coordinates

    # Spatial random effect
    s_random1 = Input(shape=(None,1))
    s_random2 = Input(shape=(None,1))
    s_random3 = Input(shape=(None,1))
    s_random4 = Input(shape=(None,1))

    # Covariate inputs
    cov1 = Input(shape=(148,1))
    cov2 = Input(shape=(148,1))


    # beta1 = Dense(256, activation='relu')(s_input)
    # beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(128, activation='relu')(s_input)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(64, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(32, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(16, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(8, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(1, activation='linear', name='beta1')(beta1)

    # beta2 = Dense(256, activation='relu')(s_input)
    # beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(128, activation='relu')(s_input)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(64, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(32, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(16, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(8, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(1, activation='linear', name='beta2')(beta2)
    # beta2 = BatchNormalization()(beta2)


    # raw_log_sigma2 = tf.Variable(initial_value=0.0, trainable=True, name="raw_log_sigma2")
    # log_sigma2 = tf.keras.activations.softplus(raw_log_sigma2)


    # Calculating y(s) using the covariates and the estimated beta functions
    y_s = Add(name='y_s')([
        Multiply()([cov1, beta1]),
        Multiply()([cov2, beta2])
    ])

    # spatial_effect = concatenate([s_random1, s_random2, s_random3,s_random4], name='spatial_concat')
    spatial_random = Dense(64, activation='relu')(s_random1)
    spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    # spatial_random = Dense(64, activation='relu')(spatial_random)
    # spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    # spatial_random = Dense(32, activation='relu')(spatial_random)
    # spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    spatial_random = Dense(32, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    spatial_random = Dense(16, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    spatial_random = Dense(8, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    spatial_random = Dense(1, activation='linear')(spatial_random)
    # spatial_random = tf.keras.layers.GlobalAveragePooling1D()(spatial_random)
    #
    spatial_random2 = Dense(64, activation='relu')(s_random2)
    spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(64, activation='relu')(spatial_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(32, activation='relu')(spatial_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    spatial_random2 = Dense(32, activation='relu')(spatial_random2)
    spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    spatial_random2 = Dense(16, activation='relu')(spatial_random2)
    spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    spatial_random2 = Dense(8, activation='relu')(spatial_random2)
    spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    spatial_random2 = Dense(1, activation='linear')(spatial_random2)
    # spatial_random2 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random2)
    #
    spatial_random3 = Dense(64, activation='relu')(s_random3)
    spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(64, activation='relu')(spatial_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(32, activation='relu')(spatial_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    spatial_random3 = Dense(32, activation='relu')(spatial_random3)
    spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    spatial_random3 = Dense(16, activation='relu')(spatial_random3)
    spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    spatial_random3 = Dense(8, activation='relu')(spatial_random3)
    spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    spatial_random3 = Dense(1, activation='linear')(spatial_random3)
    # spatial_random3 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random3)
    # #
    spatial_random4 = Dense(64, activation='relu')(s_random4)
    spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(64, activation='relu')(spatial_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(32, activation='relu')(spatial_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    spatial_random4 = Dense(32, activation='relu')(spatial_random4)
    spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    spatial_random4 = Dense(16, activation='relu')(spatial_random4)
    spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    spatial_random4 = Dense(8, activation='relu')(spatial_random4)
    spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    spatial_random4 = Dense(1, activation='linear')(spatial_random4)
    # spatial_random4 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random4)
    #
    # # bias = tf.Variable(initial_value=0.0, trainable=True)  # Bias (intercept)
    #
    # # spatial_random = tf.keras.layers.UpSampling1D(size=12)(spatial_random)

    # Add y_s with a fixed weight of 1
    # final = Add(name='final_output')([y_s,spatial_random])
    final = Add(name='final_output')([y_s,spatial_random,spatial_random2,spatial_random3,spatial_random4])


    learned_output = final



    model = Model(inputs=[s_input, cov1, cov2, s_random1, s_random2,s_random3,s_random4], outputs=learned_output)
    # model = Model(inputs=[s_input, cov1, cov2,s_random1, s_random2,s_random3,s_random4], outputs=final)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=2000,
    #     decay_rate=0.98
    # )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,  # 초기 Learning Rate 증가
        decay_steps=10000,
        decay_rate=0.98
    )

    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # model.compile(optimizer=Adam(learning_rate=0.005), loss='mse', metrics=['mse'])

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.85, beta_2=0.995)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])


    return model


def get_model_spatialmodelv4(mc=False, act="relu"):
    # Spatial coordinates input
    s_input = Input(shape=(148,3), name='spatial_coordinates')  # 3D spatial coordinates

    # Spatial random effect
    s_random1 = Input(shape=(None,1))
    s_random2 = Input(shape=(None,1))
    s_random3 = Input(shape=(None,1))
    s_random4 = Input(shape=(None,1))

    # Covariate inputs
    cov1 = Input(shape=(148,1))
    cov2 = Input(shape=(148,1))


    beta1 = Dense(128, activation='relu')(s_input)
    beta1 = get_dropout(beta1, p=0.2, mc=mc)
    # beta1 = Dense(128, activation='relu')(beta1)
    # beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(64, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.1, mc=mc)
    beta1 = Dense(32, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.2, mc=mc)
    beta1 = Dense(16, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.2, mc=mc)
    beta1 = Dense(8, activation='relu')(beta1)
    beta1 = get_dropout(beta1, p=0.2, mc=mc)
    beta1 = Dense(1, activation='linear', name='beta1')(beta1)

    beta2 = Dense(128, activation='relu')(s_input)
    beta2 = get_dropout(beta2, p=0.2, mc=mc)
    # beta2 = Dense(128, activation='relu')(beta2)
    # beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(64, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.1, mc=mc)
    beta2 = Dense(32, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.2, mc=mc)
    beta2 = Dense(16, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.2, mc=mc)
    beta2 = Dense(8, activation='relu')(beta2)
    beta2 = get_dropout(beta2, p=0.2, mc=mc)
    beta2 = Dense(1, activation='linear', name='beta2')(beta2)
    # beta2 = BatchNormalization()(beta2)


    # raw_log_sigma2 = tf.Variable(initial_value=0.0, trainable=True, name="raw_log_sigma2")
    # log_sigma2 = tf.keras.activations.softplus(raw_log_sigma2)


    # Calculating y(s) using the covariates and the estimated beta functions
    y_s = Add(name='y_s')([
        Multiply()([cov1, beta1]),
        Multiply()([cov2, beta2])
    ])

    spatial_effect = concatenate([s_random1, s_random2, s_random3, s_random4], name='spatial_concat')
    spatial_random = Dense(64, activation='relu')(spatial_effect)
    spatial_random = get_dropout(spatial_random, p=0.15, mc=mc)
    # spatial_random = Dense(64, activation='relu')(spatial_random)
    # spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    # spatial_random = Dense(32, activation='relu')(spatial_random)
    # spatial_random = get_dropout(spatial_random, p=0.1, mc=mc)
    spatial_random = Dense(32, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.15, mc=mc)
    spatial_random = Dense(16, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.15, mc=mc)
    spatial_random = Dense(8, activation='relu')(spatial_random)
    spatial_random = get_dropout(spatial_random, p=0.15, mc=mc)
    spatial_random = Dense(1, activation='linear')(spatial_random)
    # spatial_random = tf.keras.layers.GlobalAveragePooling1D()(spatial_random)
    #
    # spatial_random2 = Dense(64, activation='relu')(s_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # # spatial_random2 = Dense(64, activation='relu')(spatial_random2)
    # # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # # spatial_random2 = Dense(32, activation='relu')(spatial_random2)
    # # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(32, activation='relu')(spatial_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(16, activation='relu')(spatial_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(8, activation='relu')(spatial_random2)
    # spatial_random2 = get_dropout(spatial_random2, p=0.1, mc=mc)
    # spatial_random2 = Dense(1, activation='linear')(spatial_random2)
    # # spatial_random2 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random2)
    # #
    # spatial_random3 = Dense(64, activation='relu')(s_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # # spatial_random3 = Dense(64, activation='relu')(spatial_random3)
    # # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # # spatial_random3 = Dense(32, activation='relu')(spatial_random3)
    # # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(32, activation='relu')(spatial_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(16, activation='relu')(spatial_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(8, activation='relu')(spatial_random3)
    # spatial_random3 = get_dropout(spatial_random3, p=0.1, mc=mc)
    # spatial_random3 = Dense(1, activation='linear')(spatial_random3)
    # # spatial_random3 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random3)
    # # #
    # spatial_random4 = Dense(64, activation='relu')(s_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # # spatial_random4 = Dense(64, activation='relu')(spatial_random4)
    # # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # # spatial_random4 = Dense(32, activation='relu')(spatial_random4)
    # # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(32, activation='relu')(spatial_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(16, activation='relu')(spatial_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(8, activation='relu')(spatial_random4)
    # spatial_random4 = get_dropout(spatial_random4, p=0.1, mc=mc)
    # spatial_random4 = Dense(1, activation='linear')(spatial_random4)
    # # spatial_random4 = tf.keras.layers.GlobalAveragePooling1D()(spatial_random4)
    #
    # # bias = tf.Variable(initial_value=0.0, trainable=True)  # Bias (intercept)
    #
    # # spatial_random = tf.keras.layers.UpSampling1D(size=12)(spatial_random)

    # Add y_s with a fixed weight of 1
    final = Add(name='final_output')([y_s,spatial_random])


    learned_output = final



    model = Model(inputs=[s_input, cov1, cov2, s_random1, s_random2,s_random3,s_random4], outputs=learned_output)
    # model = Model(inputs=[s_input, cov1, cov2,s_random1, s_random2,s_random3,s_random4], outputs=final)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=2000,
    #     decay_rate=0.98
    # )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,  # 초기 Learning Rate 증가
        decay_steps=5000,
        decay_rate=0.98
    )

    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # model.compile(optimizer=Adam(learning_rate=0.005), loss='mse', metrics=['mse'])

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.85, beta_2=0.995)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])


    return model



if __name__ == '__main__':
    start_time = time.time()

    # uu = pd.read_csv('data_update/abcd_half_uu.csv')
    # covariates_rv_tmp = pd.read_csv('data_update/abcd_half.csv')



    # dataset = pd.read_csv('data_update/abcd_final_fullnet.csv')
    # dataset = pd.read_csv('data_update/abcd_final_subnet_d3_norm_scaled_y_v3.csv')
    # dataset = pd.read_csv('data_update/abcd_final_subnet_d3_norm_scaled_y_v4_update.csv')
    # dataset = pd.read_csv('data_update/abcd_final_subnet_d3_all_2.csv')
    dataset = pd.read_csv('data_update/abcd_final_subnet_d3_all_3_small_fullnetwork_700.csv')
    # 고유한 사람 ID와 ROI 추출
    unique_subjects = dataset['id'].unique()
    unique_rois = dataset['common.roi'].unique()
    unique_cate = dataset['cate'].unique()


    # scaler = StandardScaler()
    # dataset[['uu_x','uu_y','uu_z','theta0']]  = scaler.fit_transform(dataset[['uu_x','uu_y','uu_z','theta0']])

    # 초기 변수
    n_subjects = len(unique_subjects)  # 4160
    n_rois =  len(unique_rois)  # 148
    n_cate = len(unique_cate)
    n_coords = 3  # x.mni, y.mni, z.mni


    # Step 2: 3차원 배열 초기화
    output_array = np.full((n_subjects, n_rois, n_coords), np.nan, dtype=float)
    output = np.full((n_subjects, n_rois), np.nan, dtype=float)
    cov1 = np.full((n_subjects, n_rois), np.nan, dtype=float)
    cov2 = np.full((n_subjects, n_rois), np.nan, dtype=float)
    uu1 = np.full((n_subjects, n_rois), np.nan, dtype=float)
    uu2 = np.full((n_subjects, n_rois), np.nan, dtype=float)
    uu3 = np.full((n_subjects, n_rois), np.nan, dtype=float)
    uu4 = np.full((n_subjects, n_rois), np.nan, dtype=float)


    # Step 3: 데이터 채우기
    # subject와 ROI에 대한 인덱스 매핑 생성
    subject_idx_map = {subject: idx for idx, subject in enumerate(unique_subjects)}
    roi_idx_map = {roi: idx for idx, roi in enumerate(unique_rois)}
    cate_idx_map = {cate: idx for idx, cate in enumerate(unique_cate)}  # 새로운 카테고리 매핑 추가


    # 반복문을 사용해 배열에 데이터 저장
    for _, row in dataset.iterrows():
        subject = row['id']
        roi = row['common.roi']   # common.roi, cate
        cate = row['cate']  # 새로운 카테고리

        if subject in subject_idx_map and roi in roi_idx_map:
            subject_idx = subject_idx_map[subject]
            roi_idx = roi_idx_map[roi]
            # cate_idx = cate_idx_map[cate]

            output_array[subject_idx, roi_idx, 0] = row['x.mni']
            output_array[subject_idx, roi_idx, 1] = row['y.mni']
            output_array[subject_idx, roi_idx, 2] = row['z.mni']
            output[subject_idx, roi_idx] = row['logy']
            cov1[subject_idx, roi_idx] = row['gwmri']
            cov2[subject_idx, roi_idx] = row['ct']

            uu1[subject_idx, roi_idx] = row['uu_x']
            uu2[subject_idx, roi_idx] = row['theta0']
            uu3[subject_idx, roi_idx] = row['uu_y']
            uu4[subject_idx, roi_idx] = row['uu_z']
            # uu5[subject_idx, roi_idx] = row['uu_w']
            # uu6[subject_idx, roi_idx] = row['uu_v']
    # # Step 4: 배열 출력 확인
    # print(uu3.shape)  # (4160, 148, 3)
    # print(uu3)
    # sys.exit(1)


    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    roi_id_array = np.tile(np.arange(n_rois), (n_subjects, 1))


    output_train, output_temp, coords_train, coords_temp, cov1_train, cov1_temp, cov2_train, cov2_temp, uu1_train, uu1_temp, uu2_train, uu2_temp,uu3_train, uu3_temp,uu4_train, uu4_temp, = train_test_split(
        output, output_array, cov1, cov2, uu1, uu2, uu3, uu4, test_size=(val_ratio + test_ratio),
        random_state=10
    )

    # Split temp into validation and test sets
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    output_val, output_test, coords_val, coords_test, cov1_val, cov1_test, cov2_val, cov2_test, uu1_val, uu1_test, uu2_val, uu2_test,uu3_val, uu3_test,uu4_val, uu4_test = train_test_split(
        output_temp, coords_temp, cov1_temp, cov2_temp, uu1_temp, uu2_temp, uu3_temp, uu4_temp, test_size=val_test_ratio, random_state=42
    )


    roi_means = np.mean(output_train, axis=0)  # (148,)
    roi_sds = np.std(output_train, axis=0, ddof=0)  # shape: (148,)

    # 표준화 (z-score) 수행
    output_train_standardized = (output_train - roi_means) / roi_sds
    output_val_standardized = (output_val - roi_means) / roi_sds
    output_test_standardized = (output_test - roi_means) / roi_sds


    roi_means_cov1 = np.mean(cov1_train, axis=0)  # (148,)
    roi_sds_cov1 = np.std(cov1_train, axis=0, ddof=0)  # shape: (148,)


    cov1_train_standardized = (cov1_train - roi_means_cov1) / roi_sds_cov1
    cov1_val_standardized = (cov1_val - roi_means_cov1) / roi_sds_cov1
    cov1_test_standardized = (cov1_test - roi_means_cov1) / roi_sds_cov1


    roi_means_cov2 = np.mean(cov2_train, axis=0)  # (148,)
    roi_sds_cov2 = np.std(cov2_train, axis=0, ddof=0)  # shape: (148,)

    cov2_train_standardized = (cov2_train - roi_means_cov2) / roi_sds_cov2
    cov2_val_standardized = (cov2_val - roi_means_cov2) / roi_sds_cov2
    cov2_test_standardized = (cov2_test - roi_means_cov2) / roi_sds_cov2



    # #
    # save_np(cov1_train_standardized, f'data_update/cov1_train_all.ny')
    # save_np(cov1_test_standardized, f'data_update/cov1_test_all.ny')
    #
    # save_np(cov2_train_standardized, f'data_update/cov2_train_all.ny')
    # save_np(cov2_test_standardized, f'data_update/cov2_test_all.ny')
    #
    # save_np(uu1_train, f'data_update/uu1_train_all.ny')
    # save_np(uu1_test, f'data_update/uu1_test_all.ny')
    #
    # save_np(uu2_train, f'data_update/uu2_train_all.ny')
    # save_np(uu2_test, f'data_update/uu2_test_all.ny')
    #
    # save_np(uu3_train, f'data_update/uu3_train_all.ny')
    # save_np(uu3_test, f'data_update/uu3_test_all.ny')
    #
    # save_np(uu4_train, f'data_update/uu4_train_all.ny')
    # save_np(uu4_test, f'data_update/uu4_test_all.ny')
    #
    # save_np(coords_train, f'data_update/coords_train_all.ny')
    # save_np(coords_test, f'data_update/coords_test_all.ny')
    #
    # save_np(output_train_standardized, f'data_update/output_train_standardized_all.ny')
    # save_np(output_test_standardized, f'data_update/output_test_standardized_all.ny')
    #

    # save_np(cov1_train_standardized, f'data_update/cov1_train_2.ny')
    # save_np(cov1_test_standardized, f'data_update/cov1_test_2.ny')
    #
    # save_np(cov2_train_standardized, f'data_update/cov2_train_2.ny')
    # save_np(cov2_test_standardized, f'data_update/cov2_test_2.ny')
    #
    # save_np(uu1_train, f'data_update/uu1_train_2.ny')
    # save_np(uu1_test, f'data_update/uu1_test_2.ny')
    #
    # save_np(uu2_train, f'data_update/uu2_train_2.ny')
    # save_np(uu2_test, f'data_update/uu2_test_2.ny')
    #
    # save_np(uu3_train, f'data_update/uu3_train_2.ny')
    # save_np(uu3_test, f'data_update/uu3_test_2.ny')
    #
    # save_np(uu4_train, f'data_update/uu4_train_2.ny')
    # save_np(uu4_test, f'data_update/uu4_test_2.ny')
    #
    # save_np(coords_train, f'data_update/coords_train_2.ny')
    # save_np(coords_test, f'data_update/coords_test_2.ny')
    #
    # save_np(output_train_standardized, f'data_update/output_train_standardized_2.ny')
    # save_np(output_test_standardized, f'data_update/output_test_standardized_2.ny')

    #
    #
    # sys.exit(1)
    mc_model = get_model_spatialmodelv3(mc=True, act="relu")

    batch_size = 64
    epochs = 500



    h_mc_v2 = mc_model.fit(
        [coords_train, cov1_train_standardized, cov2_train_standardized,uu1_train,uu2_train,uu3_train,uu4_train], output_train_standardized,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=([coords_val, cov1_val_standardized, cov2_val_standardized,uu1_val,uu2_val,uu3_val,uu4_val], output_val_standardized)
    )



    intermediate_model = Model(inputs=mc_model.input,
                               outputs=[mc_model.get_layer('beta1').output, mc_model.get_layer('beta2').output])


    n_model = 100  # 100번 반복
    pred_sigma = []  # 결과를 저장할 리스트

    # 결과를 저장할 빈 배열 (크기는 145 x 50)
    pred_beta_coef1 = np.zeros((148, n_model))
    pred_beta_coef2 = np.zeros((148, n_model))

    # 반복문 시작


    output_array_subset = output_array[0,:,:]
    output_array_subset = output_array_subset.reshape(1,148, 3)
    cov1_subset = cov1[0,:]
    cov1_subset = cov1_subset.reshape(1, 148)
    cov2_subset = cov2[0,:]
    cov2_subset = cov2_subset.reshape(1, 148)
    uu1_subset = uu1[0,:]
    uu1_subset = uu1_subset.reshape(1, 148)
    uu2_subset = uu2[0,:]
    uu2_subset = uu2_subset.reshape(1, 148)
    uu3_subset = uu3[0,:]
    uu3_subset = uu3_subset.reshape(1, 148)
    uu4_subset = uu4[0,:]
    uu4_subset = uu4_subset.reshape(1, 148)
    # uu5_subset = uu5[0,:]
    # uu5_subset = uu5_subset.reshape(1, 148)

    # print()
    # sys.exit(1)

    # print(np.unique(coords_test, axis=0))

    for i in range(n_model):
        predicted_test = mc_model.predict([coords_test, cov1_test_standardized, cov2_test_standardized,uu1_test,uu2_test,uu3_test,uu4_test])
        pred_sigma.append(predicted_test)

        beta1_output, beta2_output = intermediate_model.predict([output_array_subset, cov1_subset, cov2_subset,uu1_subset,uu2_subset,uu3_subset,uu4_subset])


        # 예측 값을 각 반복의 열에 저장
        pred_beta_coef1[:, i] = beta1_output.flatten()
        pred_beta_coef2[:, i] = beta2_output.flatten()



    # 최종 출력 크기 확인
    print("pred_beta_coef1 shape:", pred_beta_coef1.shape)
    print("pred_beta_coef2 shape:", pred_beta_coef2.shape)

    save_np(pred_beta_coef1, f'output_beta1_abcd_real_subnet_d3_norm_scaled_y_3_small_network_700.ny')  #full: output_beta1_abcd_real_fullnet / output_beta2_abcd_real_fullnet / output_beta1_abcd_real_subnet_d3_v4
    save_np(pred_beta_coef2, f'output_beta2_abcd_real_subnet_d3_norm_scaled_y_3_small_network_700.ny')

    save_np(pred_sigma, f'prediction_abcd_real_subnet_d3_norm_scaled_y_3_small_network_700.ny')
    save_np(output_test_standardized, f'output_true_test_abcd_real_subnet_d3_norm_scaled_y_3_update_small_network_700.ny')
    save_np(coords_test, f'coords_test_abcd_real_subnet_d3_norm_scaled_y_3_update_small_network_700.ny')

    #final


    end_time = time.time()
    # 걸린 시간 출력
    print(f"Execution time: {end_time - start_time:.4f} seconds")







