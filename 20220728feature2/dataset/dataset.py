# encoding=utf8
# Copyright (c) 2022 Circue Authors. All Rights Reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import warnings
import pickle
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

def interpolate_nan_1d(arr):
    arr = np.array(arr)
    not_ok = np.isnan(arr)
    ok = np.logical_not(not_ok)
    xp = ok.ravel().nonzero()[0]
    fp = arr[ok]
    x  = not_ok.ravel().nonzero()[0]
    arr[not_ok] = np.interp(x, xp, fp)
    return arr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
from preprocess.preprocess import Preprocess
warnings.filterwarnings('ignore')


class Dataset:
    """
    数据集类
    """
    def __init__(self, args, regression_type):
        """
        初始化
        :param args: 初始化信息
        :param regression_type: 回归模型类型，共有三个回归模型类型 ["full", "variance", "discharge"]
        """
        self.args = args
        self.regression_type = regression_type
        self.all_batches_dict = None

    def load_batches_to_dict(self, bat_dict1, bat_dict2, bat_dict3):
        """
        构建字典，用来记录三个数据集的信息
        """

        batches_dict = {}

        # Replicating Load Data logic
        print("Loading batches ...")

        batch1 = bat_dict1
        # remove batteries that do not reach 80% capacity
        del batch1['b1c8']
        del batch1['b1c10']
        del batch1['b1c12']
        del batch1['b1c13']
        del batch1['b1c22']

        # updates/replaces the values of dictionary with the new dictionary)
        batches_dict.update(batch1)

        batch2 = bat_dict2
        # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
        # and put it with the correct cell from batch1
        batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
        batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
        add_len = [662, 981, 1060, 208, 482]

        for i, bk in enumerate(batch1_keys):
            batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
            for j in batch1[bk]['summary'].keys():
                if j == 'cycle':
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j],
                                                          batch2[batch2_keys[i]]['summary'][j] + len(
                                                              batch1[bk]['summary'][j])))
                else:
                    batch1[bk]['summary'][j] = np.hstack(
                        (batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
            last_cycle = len(batch1[bk]['cycles'].keys())
            for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
                batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

        del batch2['b2c7']
        del batch2['b2c8']
        del batch2['b2c9']
        del batch2['b2c15']
        del batch2['b2c16']

        # All keys have to be updated after the reordering.
        batches_dict.update(batch1)
        batches_dict.update(batch2)

        batch3 = bat_dict3
        # remove noisy channels from batch3
        del batch3['b3c37']
        del batch3['b3c2']
        del batch3['b3c23']
        del batch3['b3c32']
        del batch3['b3c38']
        del batch3['b3c39']

        batches_dict.update(batch3)

        print("Done loading batches")
        return batches_dict

    def build_feature_df(self, batch_dict):
        """
        建立一个DataFrame，包含加载的批处理字典中所有最初使用的特性
        """

        print("Start building features ...")

        # 124 cells (3 batches)
        n_cells = len(batch_dict.keys())

        ## Initializing feature vectors:
        # numpy vector with 124 zeros
        cycle_life = np.zeros(n_cells)
        # 1. delta_Q_100_10(V)
        minimum_dQ_100_10 = np.zeros(n_cells)
        variance_dQ_100_10 = np.zeros(n_cells)
        skewness_dQ_100_10 = np.zeros(n_cells)
        kurtosis_dQ_100_10 = np.zeros(n_cells)

        dQ_100_10_2 = np.zeros(n_cells)
        # 2. Discharge capacity fade curve features
        slope_lin_fit_2_100 = np.zeros(
            n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
        intercept_lin_fit_2_100 = np.zeros(
            n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
        discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
        diff_discharge_capacity_max_2 = np.zeros(n_cells)  # Difference between max discharge capacity and cycle 2
        discharge_capacity_100 = np.zeros(n_cells)  # for Fig. 1.e
        slope_lin_fit_95_100 = np.zeros(n_cells)  # for Fig. 1.f
        # 3. Other features
        mean_charge_time_2_6 = np.zeros(n_cells)  # Average charge time, cycle 1 to 5
        minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance

        diff_IR_100_2 = np.zeros(n_cells)  # Internal resistance, difference between cycle 100 and cycle 2
        temperature_integral = np.zeros(n_cells)  # Integral of cell surface temperatures over time in cycles 1-100 incl.
        temperature_intvar = np.zeros(n_cells)  # Integral variance of cell surface temperatures in cycles 1-100 incl.
        temperature_var_100_10 = np.zeros(n_cells)  # Variance of temperature difference between cycle 10 and 100
        inv_current_square_integral = np.zeros(n_cells)  # Integral of squared current (charge+discharge) over timein cycles 1-100 incl.
        dQdV_sumdiff_100_10 = np.zeros(n_cells)  # Sum of dQdV difference between cycles 100 and 10

        # Classifier features
        minimum_dQ_5_4 = np.zeros(n_cells)
        variance_dQ_5_4 = np.zeros(n_cells)
        cycle_550_clf = np.zeros(n_cells)

        # iterate/loop over all cells.
        for i, cell in enumerate(batch_dict.values()):
            cycle_life[i] = cell['cycle_life']
            # 1. delta_Q_100_10(V)
            c10 = cell['cycles']['10']
            c100 = cell['cycles']['100']
            dQ_100_10 = c100['Qdlin'] - c10['Qdlin']

            minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))
            variance_dQ_100_10[i] = np.log(np.abs(np.var(dQ_100_10)))
            skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
            kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

            Qdlin_100_10 = cell['cycles']['100']['Qdlin'] - cell['cycles']['10']['Qdlin']
            dQ_100_10_2[i] = np.var(Qdlin_100_10)

            # 2. Discharge capacity fade curve features
            # Compute linear fit for cycles 2 to 100:
            q = cell['summary']['QD'][1:100].reshape(-1, 1)  # discharge capacities; q.shape = (99, 1);
            X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
            linear_regressor_2_100 = LinearRegression()
            linear_regressor_2_100.fit(X, q)

            slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
            intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
            discharge_capacity_2[i] = q[0][0]
            diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

            discharge_capacity_100[i] = q[-1][0]

            q95_100 = cell['summary']['QD'][94:100].reshape(-1, 1)
            q95_100 = q95_100 * 1000  # discharge cappacities; q.shape = (99, 1);
            X95_100 = cell['summary']['cycle'][94:100].reshape(-1,
                                                               1)  # Cylce index from 2 to 100; X.shape = (99, 1)
            linear_regressor_95_100 = LinearRegression()
            linear_regressor_95_100.fit(X95_100, q95_100)
            slope_lin_fit_95_100[i] = linear_regressor_95_100.coef_[0]

            # 3. Other features
            mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
            minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
            diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]
            temperature_integral_cycle = np.zeros(100-1)
            current_square_integral_cycle = np.zeros(100-1)
            for j in range(1, 100):
                cycle_data = cell['cycles'][str(j)]
                timestep = cycle_data['t'][1:] - cycle_data['t'][:-1]
                timestep = np.minimum(timestep, np.median(timestep) * 30.)
                prod_T_t = (cycle_data['T'][1:] + cycle_data['T'][:-1]) / 2 * timestep
                temperature_integral_cycle[j-1] = np.sum(prod_T_t)
                prod_curr2_t = ((cycle_data['I'][1:] + cycle_data['I'][:-1]) / 2) ** 2 * timestep
                current_square_integral_cycle[j-1] = np.sum(prod_curr2_t)
            temperature_integral[i] = np.sum(temperature_integral_cycle)
            inv_current_square_integral[i] = 1e5 / np.sum(current_square_integral_cycle)
            temperature_intvar[i] = np.log10(np.var(temperature_integral_cycle))

            cycle_end = '100'
            cycle_start = '10'
            vol2timefun = interp1d(cell['cycles'][cycle_end]['V'], cell['cycles'][cycle_end]['t'])
            time2Tempfun = interp1d(cell['cycles'][cycle_end]['t'], cell['cycles'][cycle_end]['T'])
            voltage_base = np.linspace(2.0, 3.6, 500)
            Temp100 = time2Tempfun(vol2timefun(voltage_base))
            vol2timefun = interp1d(cell['cycles'][cycle_start]['V'], cell['cycles'][cycle_start]['t'])
            time2Tempfun = interp1d(cell['cycles'][cycle_start]['t'], cell['cycles'][cycle_start]['T'])
            voltage_base = np.linspace(2.0, 3.6, 500)
            Temp10 = time2Tempfun(vol2timefun(voltage_base))
            temperature_var_100_10[i] = np.log10(np.var(Temp100 - Temp10))
            dQdV_sumdiff_100_10[i] = np.sum(cell['cycles'][cycle_end]['dQdV'] - cell['cycles'][cycle_start]['dQdV'])

            # Classifier features
            c4 = cell['cycles']['4']
            c5 = cell['cycles']['5']
            dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
            minimum_dQ_5_4[i] = np.log10(np.abs(np.min(dQ_5_4)))
            variance_dQ_5_4[i] = np.log10(np.var(dQ_5_4))
            cycle_550_clf[i] = cell['cycle_life'] >= 550

        # slope_lin_fit_2_100_ = 1. / (0.1 + np.exp(-(slope_lin_fit_2_100 + 0.0001) * 20000)) + slope_lin_fit_2_100 * 2000
        # slope_lin_fit_95_100_ = 0. / (1 + np.exp(-20 * slope_lin_fit_95_100 - 1)) + slope_lin_fit_95_100 * 20

        # combining all featues in one big matrix where rows are the cells and colums are the features
        # note the last 2 variables below are labels/targets for ML i.e cycle life and cycle_550_clf
        features_df = pd.DataFrame({
            "cell_key": np.array(list(batch_dict.keys())),                      # 0
            "minimum_dQ_100_10": minimum_dQ_100_10,                             # 1
            "variance_dQ_100_10": variance_dQ_100_10,                           # 2
            "skewness_dQ_100_10": skewness_dQ_100_10,                           # 3
            "kurtosis_dQ_100_10": kurtosis_dQ_100_10,                           # 4
            "slope_lin_fit_2_100": slope_lin_fit_2_100,                         # 5
            "intercept_lin_fit_2_100": intercept_lin_fit_2_100,                 # 6
            "discharge_capacity_2": discharge_capacity_2,                       # 7
            "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,     # 8
            "mean_charge_time_2_6": mean_charge_time_2_6,                       # 9
            "minimum_IR_2_100": minimum_IR_2_100,                               # 10
            "diff_IR_100_2": diff_IR_100_2,                                     # 11
            "minimum_dQ_5_4": minimum_dQ_5_4,                                   # 12
            "variance_dQ_5_4": variance_dQ_5_4,                                 # 13
            "temperature_integral":temperature_integral,                        # 14
            "temperature_intvar":temperature_intvar,                            # 15
            "slope_lin_fit_95_100":slope_lin_fit_95_100,                        # 16
            "temperature_var_100_10":temperature_var_100_10,                    # 17
            "inv_current_square_integral":inv_current_square_integral,          # 18
            "dQdV_variance_100_10":dQdV_sumdiff_100_10,                        # 19
            "cycle_life": cycle_life,                                           # -2
            "cycle_550_clf": cycle_550_clf,                                     # -1
        })

        print("Done building features")
        return features_df

    def train_val_split(self, features_df, regression_type="full", model="regression"):
        """
        划分train&test数据集。注意：数据集要按照指定方式划分
        :param features_df: 包含最初使用的特性dataframe
        :param regression_type: 回归模型的类型
        :param model: 使用模型的flag
        """
        # only three+1 versions are allowed.
        assert regression_type in ["full", "variance", "discharge", "mycomb"]

        # dictionary to hold the features indices for each model version.
        features = {
            "full": [1, 2, 5, 6, 7, 9, 10, 11, 14],
            "variance": [2],
            "discharge": [1, 2, 3, 4, 7, 8],
            "mycomb": [1, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 19],
        }
        # get the features for the model version (full, variance, discharge)
        feature_indices = features[regression_type]
        # get all cells with the specified features
        model_features = features_df.iloc[:, feature_indices]
        # get last two columns (cycle life and classification)
        labels = features_df.iloc[:, -2:]
        # labels are (cycle life ) for regression other wise (0/1) for classsification
        labels = labels.iloc[:, 0] if model == "regression" else labels.iloc[:, 1]

        # split data in to train/primary_test/and secondary test
        train_cells = np.arange(1, 84, 2)
        val_cells = np.arange(0, 84, 2)
        test_cells = np.arange(84, 124, 1)

        # get cells and their features of each set and convert to numpy for further computations
        x_train = np.array(model_features.iloc[train_cells])
        x_val = np.array(model_features.iloc[val_cells])
        x_test = np.array(model_features.iloc[test_cells])

        # target values or labels for training
        y_train = np.array(labels.iloc[train_cells])
        y_val = np.array(labels.iloc[val_cells])
        y_test = np.array(labels.iloc[test_cells])

        # return 3 sets
        return {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}

    def get_feature(self):
        """
        类主函数，返回可用于训练的数据集
        """
        pre_dataset = Preprocess(args = self.args)
        if self.all_batches_dict is None:
            if os.path.exists(self.args.preprocessDataFrameFilename):
                with open(self.args.preprocessDataFrameFilename, 'rb') as f:
                    self.all_batches_dict = pickle.load(f)
            else:
                bat_dict1, bat_dict2, bat_dict3 = pre_dataset.data_preprocess()
                # calling function to load from disk
                self.all_batches_dict = self.load_batches_to_dict(bat_dict1, bat_dict2, bat_dict3)
                with open(self.args.preprocessDataFrameFilename, 'wb+') as f:
                    pickle.dump(self.all_batches_dict, f)
        # function to build features for ML
        features_df = self.build_feature_df(self.all_batches_dict)
        battery_dataset = self.train_val_split(features_df, self.regression_type)

        return battery_dataset


