#ifndef __SOLVERESULT_HH__
#define __SOLVERESULT_HH__

#include <vector>
#include "eigen/Eigen/Dense"

struct SolveResult {
    Eigen::VectorXd solution;
    int iterations;
    double time;
    std::vector<double> errors;
};

#endif