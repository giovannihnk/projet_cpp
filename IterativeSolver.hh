#ifndef __ITERATIVESOLVER_HH__
#define __ITERATIVESOLVER_HH__

#include "eigen/Eigen/Sparse"
#include "SolveResult.hh"

class IterativeSolver {
protected:
    double _tol;
    int _max_iter;

public:
    IterativeSolver(double tol = 1e-6, int max_iter = 1000)
        : _tol(tol), _max_iter(max_iter) {}

    virtual ~IterativeSolver() = default;

    virtual SolveResult solve(const Eigen::SparseMatrix<double>& A,
                              const Eigen::VectorXd& b,
                              const Eigen::VectorXd& x0,
                              const Eigen::VectorXd& x_exact) = 0;
};

#endif
