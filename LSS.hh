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

double const tol = 1e-6;
int const max_iter = 1000 ;

class LSS{
    public :     
        // ================= Méthodes de résolution de système linéaire ==================
        tuple<VectorXd, int, double, vector<double>> jacobi_dense_with_error(const MatrixXd& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact);
        tuple<VectorXd, int, double, vector<double>> jacobi_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact);
        tuple<VectorXd, int, double, vector<double>> gauss_seidel_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact);
        tuple<VectorXd, int, double, vector<double>> SOR_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double omega);

        // ================= Fonctionnalités supplémentaires ==================
        pair<SparseMatrix<double>, MatrixXd> generate_simple_sparse_tridiagonal_matrix(int n, double diagonal_value , double off_diagonal_value);
        double rayon_spectral_JS(const MatrixXd& A);
        double rayon_spectral_GS(const MatrixXd& A);
        double rayon_spectral_SOR(const MatrixXd& A, double omega);
        void diagonale_dominante(const MatrixXd& A);

};

#endif