#include "JacobiSolver.hh"
#include <chrono>

SolveResult JacobiSolver::solve(const Eigen::SparseMatrix<double>& A,
                                const Eigen::VectorXd& b,
                                const Eigen::VectorXd& x0,
                                const Eigen::VectorXd& x_exact) {
    auto start = std::chrono::high_resolution_clock::now();
    int n = A.rows();
    Eigen::VectorXd x = x0;
    std::vector<double> errors;

    Eigen::VectorXd diag = A.diagonal();
    Eigen::VectorXd diag_inv = diag.cwiseInverse();
    Eigen::SparseMatrix<double> M = A;
    for(int i=0; i<n; i++) M.coeffRef(i,i) = 0.0;

    for(int k=0; k<_max_iter; k++) {
        Eigen::VectorXd x_new = diag_inv.asDiagonal() * (b - M * x);
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < _tol) break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration<double>(end - start).count();

    return {x, (int)errors.size(), time_taken, errors};
}
