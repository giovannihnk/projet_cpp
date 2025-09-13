#include "MatrixGenerator.hh"
#include "eigen/Eigen/Sparse"

std::pair<Eigen::SparseMatrix<double>, Eigen::MatrixXd>
MatrixGenerator::generateTridiagonal(int n, double diagonal_value, double off_diagonal_value) {
    std::vector<Eigen::Triplet<double>> triplets;

    for(int i=0; i<n; i++) {
        triplets.emplace_back(i, i, diagonal_value);
        if(i < n-1) {
            triplets.emplace_back(i, i+1, off_diagonal_value);
            triplets.emplace_back(i+1, i, off_diagonal_value);
        }
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::MatrixXd A_dense = Eigen::MatrixXd::Zero(n, n);
    for(int i=0; i<n; i++) {
        A_dense(i, i) = diagonal_value;
        if(i < n-1) {
            A_dense(i, i+1) = off_diagonal_value;
            A_dense(i+1, i) = off_diagonal_value;
        }
    }

    return {A, A_dense};
}
