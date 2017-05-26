#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to calculate NIS.
  */
  double CalculateNIS(const Eigen::VectorXd &estimation, const Eigen::VectorXd &ground_truth, const Eigen::MatrixXd &covariance_matrix);
};

#endif /* TOOLS_H_ */
