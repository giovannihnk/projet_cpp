#include "LSS.hh"
#include <iostream>
using namespace std;
using namespace Eigen;

int main() {
    int n = 10;
    double diagonal_value = 4.0;
    double off_diagonal_value = -1.0;

    LSS<MatrixXd, VectorXd> solv;

   // auto [A_sparse, A_dense] = solv.generate_simple_sparse_tridiagonal_matrix(...);
    auto matrices = solv.generate_simple_sparse_tridiagonal_matrix(n, diagonal_value, off_diagonal_value);
    SparseMatrix<double> A_sparse = matrices.first;
    MatrixXd A_dense = matrices.second;


    VectorXd x_exact = VectorXd::Ones(n);
    VectorXd b_h = A_dense * x_exact;
    VectorXd x0 = VectorXd::Zero(n);

    // Jacobi Dense
    auto resultJD = solv.jacobi_dense_with_error(A_dense, b_h, x0, x_exact);
    vector<double> errorsJD = get<3>(resultJD);
    cout << "Jacobi Dense: Iterations = " << get<1>(resultJD)
         << ", Time = " << get<2>(resultJD) << "s, Final Error = " << errorsJD.back() << endl;

    // Jacobi Sparse
    auto resultJS = solv.jacobi_sparse_with_error(A_sparse, b_h, x0, x_exact);
    vector<double> errorsJS = get<3>(resultJS);
    cout << "Jacobi Sparse: Iterations = " << get<1>(resultJS)
         << ", Time = " << get<2>(resultJS) << "s, Final Error = " << errorsJS.back() << endl;

    // Gauss-Seidel Sparse
    auto resultGS = solv.gauss_seidel_sparse_with_error(A_sparse, b_h, x0, x_exact);
    vector<double> errorsGS = get<3>(resultGS);
    cout << "Gauss-Seidel Sparse: Iterations = " << get<1>(resultGS)
         << ", Time = " << get<2>(resultGS) << "s, Final Error = " << errorsGS.back() << endl;

    // SOR Sparse
    auto resultSOR = solv.SOR_sparse_with_error(A_sparse, b_h, x0, x_exact, 1.8);
    vector<double> errorsSOR = get<3>(resultSOR);
    cout << "SOR Sparse: Iterations = " << get<1>(resultSOR)
         << ", Time = " << get<2>(resultSOR) << "s, Final Error = " << errorsSOR.back() << endl;

    return 0;
}
