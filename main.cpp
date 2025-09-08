#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <chrono>
#include <cmath>
#include "LSS.hh"

using namespace Eigen;
using namespace std;

int main() {
    LSS<MatrixXd, VectorXd> solv;  // instanciation correcte
    vector<int> tailles = {10,20,30};
    for(int n : tailles) {
        VectorXd x0 = VectorXd::Zero(n);
        auto matrices = solv.generate_simple_sparse_tridiagonal_matrix(n, -1, 2);
        SparseMatrix<double> A_sparse = matrices.first;
        MatrixXd A_dense = matrices.second;

        double r1 = solv.rayon_spectral_JS(A_dense);
        double r2 = solv.rayon_spectral_GS(A_dense);
        double r3 = solv.rayon_spectral_SOR(A_dense, 1.8);

        cout << "Rayon Jacobi pour " << n << " : " << r1 << endl;
        cout << "Rayon GS pour " << n << " : " << r2 << endl;
        cout << "Rayon SOR pour " << n << " : " << r3 << endl;

        VectorXd b = VectorXd::Random(n);
        b(0) = 0; b(n-1) = 0;
        double h = 1.0/(n+1);
        VectorXd b_h = h*h*b;

        VectorXd x_exact = A_dense.colPivHouseholderQr().solve(b_h);

        auto resultJD = solv.jacobi_dense_with_error(A_dense, b_h, x0, x_exact);
        auto resultJS = solv.jacobi_sparse_with_error(A_sparse, b_h, x0, x_exact);
        auto resultGS = solv.gauss_seidel_sparse_with_error(A_sparse, b_h, x0, x_exact);
        auto resultSOR = solv.SOR_sparse_with_error(A_sparse, b_h, x0, x_exact, 1.8);

        cout << "Jacobi Dense itérations: " << get<1>(resultJD) << " temps: " << get<2>(resultJD) << "s\n";
        cout << "Jacobi Sparse itérations: " << get<1>(resultJS) << " temps: " << get<2>(resultJS) << "s\n";
        cout << "Gauss-Seidel itérations: " << get<1>(resultGS) << " temps: " << get<2>(resultGS) << "s\n";
        cout << "SOR itérations: " << get<1>(resultSOR) << " temps: " << get<2>(resultSOR) << "s\n";
    }
    return 0;
}
