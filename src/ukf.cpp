#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // coef = sqrt(lambda_ + n_aug_);

  // if this is false, laser measurements will be ignored (except during init)
  use_ladar_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //TODO: find the correct initial value
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  //TODO: find the correct initial value
  std_yawdd_ = 30;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<  1, 0, 0, 0,0,
        0, 1, 0, 0,0,
        0, 0, 1000, 0,0,
        0, 0, 0, 1000,0,
        0,0,0,0,1000;

  Q_ = MatrixXd(2, 2);
  Q_ << std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;

  R_radar_ = MatrixXd(n_z_radar, n_z_radar);
  R_radar_ << std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0, std_radrd_*std_radrd_;

  R_ladar_ = MatrixXd(n_z_ladar, n_z_ladar);
  R_ladar_ << std_laspx_*std_laspx_, 0,
    0, std_laspy_*std_laspy_;

  // initial sigma point prediction
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //create matrix for sigma points in measurement space
  Zsig_radar = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
  Zsig_ladar = MatrixXd(n_z_ladar, 2 * n_aug_ + 1);

  // initial weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5/(n_aug_ + lambda_);
  }

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "UKF: initializing ..." << endl;

    //set the state with the initial location and zero velocity
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      const float ro = meas_package.raw_measurements_[0];
      const float phi = meas_package.raw_measurements_[1];
      const float rho_dot = meas_package.raw_measurements_[2];
      x_ << ro * cos(phi), ro * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    return;
  }

  /*****************************************************************************
  *  Prediction
  ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  UKF::Prediction(dt);
  cout << "Prediction" << endl;

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UKF::UpdateRadar(meas_package.raw_measurements_);
    cout << "Radar Update" << endl;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_ladar_){
    UKF::UpdateLidar(meas_package.raw_measurements_);
    cout << "Ladar Update" << endl;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  // generate sigma points
  UKF::GenerateAugmentedSigmaPoints(Xsig_aug);

  //reset matrix with predicted sigma points
  Xsig_pred_.fill(0.0);
  // predict generated sigma points
  UKF::SigmaPointPrediction(Xsig_pred_, Xsig_aug, delta_t);

  // calculate mean and covariance of the predicted sigma points
  UKF::PredictMeanAndCovariance(Xsig_pred_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const VectorXd &z) {
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_ladar);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_ladar,n_z_ladar);
  UKF::PredictLadarMeasurement(z_pred, S);
  UKF::LadarUpdateStateHelper(z_pred, S, z);
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const VectorXd &z) {
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar,n_z_radar);
  UKF::PredictRadarMeasurement(z_pred, S);
  UKF::RadarUpdateStateHelper(z_pred, S, z);
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

/***************************************************************************
*  Helper functions                                                        *
****************************************************************************/
void UKF::GenerateAugmentedSigmaPoints(MatrixXd& Xsig_aug) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(5) = x_;
  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_-n_x_,n_aug_-n_x_) = Q_;
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd A_Scaled = coef*A;

  //set first column of sigma point matrix
  Xsig_aug.col(0) = x_aug;
  //set remaining sigma points
  for(int i = 0; i < n_aug_; i++) {
     Xsig_aug.col(i + 1) = x_aug + A_Scaled.col(i);
     Xsig_aug.col(i + n_aug_ + 1) = x_aug - A_Scaled.col(i);
  }

  //print result
  // std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
}

// augmented sigma point matrix
// MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
void UKF::SigmaPointPrediction(MatrixXd& Xsig_pred, const MatrixXd& Xsig_aug, const double delta_t) {
  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //print result
  // std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
}

// matrix with predicted sigma points
// MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
void UKF::PredictMeanAndCovariance(const MatrixXd& Xsig_pred) {
  //predict state mean
  x_.fill(0.0);
  P_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred.col(i);
  }

  //predict state covariance matrix
 VectorXd x_diff;
 for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_diff = Xsig_pred.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
 }

  //print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x_ << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P_ << std::endl;
}

//mean predicted measurement
// VectorXd z_pred = VectorXd(n_z_radar);
//measurement covariance matrix S
// MatrixXd S = MatrixXd(n_z_radar,n_z_radar);
void UKF::PredictRadarMeasurement(VectorXd& z_pred, MatrixXd& S) {
  Zsig_radar.fill(0.0);

  //transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    Zsig_radar(0, i) = sqrt(p_x*p_x + p_y*p_y);
    Zsig_radar(1, i) = atan2(p_y, p_x);
    Zsig_radar(2, i) = (p_x*v*cos(yaw)+p_y*v*sin(yaw))/Zsig_radar(0, i);
  }

  //calculate mean predicted measurement
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_radar.col(i);
  }

  //calculate measurement covariance matrix S
  VectorXd z_diff;
  S = R_radar_;
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_diff = Zsig_radar.col(i) - z_pred;
      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1) -= 2. *M_PI;
      while (z_diff(1)<-M_PI) z_diff(1) += 2. *M_PI;

      S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;
}

//mean predicted measurement
// VectorXd z_pred = VectorXd(n_z_ladar);
//measurement covariance matrix S
// MatrixXd S = MatrixXd(n_z_ladar,n_z_ladar);
void UKF::PredictLadarMeasurement(VectorXd& z_pred, MatrixXd& S) {
  //reset matrix for sigma points in measurement space
  Zsig_ladar.fill(0.0);

  //transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig_ladar(0, i) = Xsig_pred_(0,i); //p_x
    Zsig_ladar(1, i) = Xsig_pred_(1,i); //p_y
  }

  //calculate mean predicted measurement
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_ladar.col(i);
  }

  //calculate measurement covariance matrix S
  VectorXd z_diff;
  S = R_ladar_;
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_diff = Zsig_ladar.col(i) - z_pred;
      S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;
}

// matrix with sigma points in measurement space
// MatrixXd Zsig_radar = MatrixXd(n_z_radar, 2 * n_aug_ + 1);
// vector for mean predicted measurement
// VectorXd z_pred = VectorXd(n_z_radar);
// matrix for predicted measurement covariance
// MatrixXd S = MatrixXd(n_z_radar,n_z_radar);
// vector for incoming radar measurement
// VectorXd z = VectorXd(n_z_radar);
void UKF::RadarUpdateStateHelper(const VectorXd& z_pred, const MatrixXd& S, const VectorXd& z) {
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar);
  //create matrix for Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z_radar);

  //calculate cross correlation matrix
  VectorXd x_diff;
  VectorXd z_diff;
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      z_diff = Zsig_radar.col(i) - z_pred;
      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //calculate Kalman gain K;
  K = Tc*S.inverse();

  //update state mean and covariance matrix

  //residual
  z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();

  //print result
  // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}

// vector for mean predicted measurement
// VectorXd z_pred = VectorXd(n_z_ladar);
// matrix for predicted measurement covariance
// MatrixXd S = MatrixXd(n_z_ladar,n_z_ladar);
// vector for incoming ladar measurement
// VectorXd z = VectorXd(n_z_ladar);
void UKF::LadarUpdateStateHelper(const VectorXd& z_pred, const MatrixXd& S, const VectorXd& z) {
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_ladar);
  //create matrix for Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z_ladar);

  //calculate cross correlation matrix
  VectorXd x_diff;
  VectorXd z_diff;
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
      x_diff = Xsig_pred_.col(i) - x_;
      z_diff = Zsig_ladar.col(i) - z_pred;

      Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //calculate Kalman gain K;
  K = Tc*S.inverse();

  //update state mean and covariance matrix

  //residual
  z_diff = z - z_pred;

  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();

  //print result
  // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}
