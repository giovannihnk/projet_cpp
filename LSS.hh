#ifndef __LSS_HH__
#define __LSS_HH__

#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <chrono>
#include <cmath>

using namespace Eigen;
using namespace std;



template <typename MatrixType, typename VectorType>
class LSS{
    private:
        double _tol;
        int _max_iter;
        
    public :     
        LSS();
        void setTol(double tol);
        void setMaxIter(int iter);
        double getTol();
        int getMaxIter();
        tuple<VectorType, int, double, vector<double>> jacobi_dense_with_error(const MatrixType& A, const VectorType& b, const VectorType& x0, const VectorType& x_exact);
        tuple<VectorType, int, double, vector<double>> jacobi_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b, const VectorType& x0, const VectorType& x_exact);
        tuple<VectorType, int, double, vector<double>> gauss_seidel_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b, const VectorType& x0, const VectorType& x_exact);
        tuple<VectorType, int, double, vector<double>> SOR_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b, const VectorType& x0, const VectorType& x_exact, double omega);
        pair<SparseMatrix<double>, MatrixType> generate_simple_sparse_tridiagonal_matrix(int n, double diagonal_value , double off_diagonal_value);
        double rayon_spectral_JS(const MatrixType& A);
        double rayon_spectral_GS(const MatrixType& A);
        double rayon_spectral_SOR(const MatrixType& A, double omega);
        void diagonale_dominante(const MatrixType& A);
};

#endif