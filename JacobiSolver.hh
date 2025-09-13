#ifndef __JACOBISOLVER_HH__
#define __JACOBISOLVER_HH__

#include "IterativeSolver.hh"

class JacobiSolver : public IterativeSolver {
public:
    using IterativeSolver::IterativeSolver;

    SolveResult solve(const Eigen::SparseMatrix<double>& A,
                      const Eigen::VectorXd& b,
                      const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& x_exact) override;
};

#endif
