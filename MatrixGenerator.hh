#ifndef __MATRIXGENERATOR_HH__
#define __MATRIXGENERATOR_HH__

#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <vector>

#include <utility>

class MatrixGenerator {
public:
    static std::pair<Eigen::SparseMatrix<double>, Eigen::MatrixXd>
    generateTridiagonal(int n, double diagonal_value, double off_diagonal_value);
};

#endif
