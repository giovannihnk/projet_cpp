#include "GaussSeidelSolver.hh"
#include <chrono>

SolveResult GaussSeidelSolver::solve(const Eigen::SparseMatrix<double>& A,
                                     const Eigen::VectorXd& b,
                                     const Eigen::VectorXd& x0,
                                     const Eigen::VectorXd& x_exact) {
    auto start = std::chrono::high_resolution_clock::now();
    int n = A.rows();
    Eigen::VectorXd x = x0;
    std::vector<double> errors;

    for(int k=0; k<_max_iter; k++) {
        Eigen::VectorXd x_new = x;
        for(int i=0; i<n; i++) {
            double s1 = 0.0, s2 = 0.0;
            for(Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if(it.col() < i) s1 += it.value() * x_new(it.col());
                else if(it.col() > i) s2 += it.value() * x(it.col());
            }
            x_new(i) = (b(i) - s1 - s2)/A.coeff(i,i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        if(error < _tol) break;
        x = x_new;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration<double>(end - start).count();

    return {x, (int)errors.size(), time_taken, errors};
}
