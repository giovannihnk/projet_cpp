#ifndef __SORSOLVER_HH__
#define __SORSOLVER_HH__

#include "IterativeSolver.hh"

class SORSolver : public IterativeSolver {
private:
    double _omega;

public:
    SORSolver(double omega, double tol = 1e-6, int max_iter = 1000)
        : IterativeSolver(tol, max_iter), _omega(omega) {}

    SolveResult solve(const Eigen::SparseMatrix<double>& A,
                      const Eigen::VectorXd& b,
                      const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& x_exact) override;
};

#endif
