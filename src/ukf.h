#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
  ///* State dimension
  const int n_x_ = 5;

  ///* Augmented state dimension
  const int n_aug_ = 7;

  //set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z_radar = 3;

  //set measurement dimension, radar can measure r, phi, and r_dot
  const int n_z_ladar = 2;

  ///* Sigma point spreading parameter
  const int lambda_ = 3 - n_aug_;

  // sqrt(lambda + n_aug)
  const double coef = sqrt(lambda_ + n_aug_);

  ///* if this is false, ladar measurements will be ignored (except for init)
  bool use_ladar_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  const double std_laspx_ = 0.15;

  ///* Laser measurement noise standard deviation position2 in m
  const double std_laspy_ = 0.15;

  ///* Radar measurement noise standard deviation radius in m
  const double std_radr_ = 0.3;

  ///* Radar measurement noise standard deviation angle in rad
  const double std_radphi_ = 0.03;

  ///* Radar measurement noise standard deviation radius change in m/s
  const double std_radrd_  = 0.3;

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;
  MatrixXd H_;
  MatrixXd Q_;
  MatrixXd R_radar_;
  MatrixXd R_ladar_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_radar;
  MatrixXd Zsig_ladar;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* the current NIS for radar
  double NIS_radar_;
  int total_radar_measurements;
  int NIS_radar_above_low;
  int NIS_radar_above_high;
  const double NIS_radar_low = 0.352;
  const double NIS_radar_high = 7.815;

  ///* the current NIS for laser
  double NIS_laser_;
  int total_ladar_measurements;
  int NIS_ladar_above_low;
  int NIS_ladar_above_high;
  const double NIS_ladar_low = 0.103;
  const double NIS_ladar_high = 5.991;

  Tools tools;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param z The measurement at k+1
   */
  void UpdateLidar(const VectorXd& z);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param z The measurement at k+1
   */
   void UpdateRadar(const VectorXd& z);

private:
  void GenerateAugmentedSigmaPoints(MatrixXd& Xsig_aug);
  void SigmaPointPrediction(MatrixXd& Xsig_pred, const MatrixXd& Xsig_aug, const double delta_t);
  // void SigmaPointPrediction(const MatrixXd& Xsig_aug, const double delta_t);
  void PredictMeanAndCovariance(const MatrixXd& Xsig_pred);
  // void PredictMeanAndCovariance();
  void PredictRadarMeasurement(VectorXd& z_pred, MatrixXd& S);
  void PredictLadarMeasurement(VectorXd& z_pred, MatrixXd& S);
  void RadarUpdateStateHelper(const VectorXd& z_pred, const MatrixXd& S, const VectorXd& z);
  void LadarUpdateStateHelper(const VectorXd& z_pred, const MatrixXd& S, const VectorXd& z);
  void NormalizeAngle(double& angle);
};

#endif /* UKF_H */
