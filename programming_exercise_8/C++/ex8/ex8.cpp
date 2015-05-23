// Copyright (C) 2015  Caleb Lo
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

// Machine Learning
// Programming Exercise 8: Anomaly Detection
// Problem: Apply anomaly detection to detect anomalous behavior in servers

#include "anomaly_detection.h"

int main(void) {
  const std::string kServerData1FileName = "../../serverData1.txt";
  DataUnlabeled server_data_1(kServerData1FileName);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Estimate (Gaussian) statistics of this dataset.
  const std::string kServerValData1FileName = "../../serverValData1.txt";
  Data server_val_data_1(kServerValData1FileName);
  AnomalyDetection anom_detect(server_data_1,server_val_data_1);
  const int kReturnCode = anom_detect.EstimateGaussian(server_data_1);
  const int kReturnCode2 = anom_detect.MultivariateGaussian(server_data_1,0);
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Use a cross-validation set to find outliers.
  const int kReturnCode3 = \
    anom_detect.MultivariateGaussian(server_val_data_1,1);
  const int kReturnCode4 = anom_detect.SelectThreshold(server_val_data_1);
  printf("Best epsilon found using cross-validation: %e\n",\
    anom_detect.best_epsilon());
  printf("Best F1 on Cross Validation Set:  %f\n",anom_detect.best_F1());
  printf("   (you should see a value epsilon of about 8.99e-05)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  // Detect anomalies in another dataset.
  const std::string kServerData2FileName = "../../serverData2.txt";
  DataUnlabeled server_data_2(kServerData2FileName);

  // Estimate (Gaussian) statistics of this dataset.
  const std::string kServerValData2FileName = "../../serverValData2.txt";
  Data server_val_data_2(kServerValData2FileName);
  AnomalyDetection anom_detect_2(server_data_2,server_val_data_2);
  const int kReturnCode5 = anom_detect_2.EstimateGaussian(server_data_2);
  const int kReturnCode6 = anom_detect_2.MultivariateGaussian(server_data_2,0);

  // Use a cross-validation set to find outliers in this dataset.
  const int kReturnCode7 = \
    anom_detect_2.MultivariateGaussian(server_val_data_2,1);
  const int kReturnCode8 = anom_detect_2.SelectThreshold(server_val_data_2);
  printf("Best epsilon found using cross-validation: %e\n",\
    anom_detect_2.best_epsilon());
  printf("Best F1 on Cross Validation Set:  %f\n",anom_detect_2.best_F1());
  std::vector<int> outlier_indices;
  for(int ex_index=0;ex_index<server_data_2.num_train_ex();ex_index++)
  {
    if (anom_detect_2.data_probs()[ex_index] < anom_detect_2.best_epsilon()) {
      outlier_indices.push_back(ex_index);
	}
  }
  printf("# Outliers found: %d\n",outlier_indices.size());
  printf("   (you should see a value epsilon of about 1.38e-18)\n");
  printf("Program paused. Press enter to continue.\n");
  std::cin.ignore();

  return 0;
}
