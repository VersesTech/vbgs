# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


def calc_mse(x, y):
    if x.max() <= 1:
        x = x * 255
    if y.max() <= 1:
        y = y * 255

    return np.mean((1.0 * x - 1.0 * y) ** 2)


def calc_psnr(x, y):
    if x.max() <= 1:
        x = x * 255
    if y.max() <= 1:
        y = y * 255

    mse_ = calc_mse(x, y)
    return 20 * np.log10(254 / np.sqrt(mse_))
