# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Concept drift example.
Download dataset from: https://www.kaggle.com/camnugent/sandp500.
File structure:
    --archive
        --individual_stocks_5yr
            --__MACOSX
            --individual_stocks_5yr
        --all_stocks_5yr.csv
        --getSandP.py
        --merge.sh
Please use the data in archive/individual_stocks_5yr/individual_stocks_5yr/XX.csv.
In each csv file, there are 'date','open','high','low','close','volume','Name' columns.
Please choose one column or multiple columns.
'date' and 'Name' are non-data column, please do not use.
"""

import numpy as np
from mindarmour import ConceptDriftCheckTimeSeries


# input data
DATA_FILE = r'archive/individual_stocks_5yr/individual_stocks_5yr/AEE_data.csv'
data = np.loadtxt(DATA_FILE, str, delimiter=",")
data = data[1:, 2].astype('float64')  # choose one column or multiple columns data[1:, 2:5]
# Initialization
concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10,
                                      step=10, threshold_index=1.5, need_label=False)
# drift check
drift_score, threshold, concept_drift_location = concept.concept_check(data)
