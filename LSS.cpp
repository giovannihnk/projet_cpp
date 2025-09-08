#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include "LSS.hh"
#include <chrono>
#include <cmath>

using namespace Eigen;
using namespace std;

template class LSS<MatrixXd, VectorXd>;


template<typename MatrixType, typename VectorType>
LSS<MatrixType, VectorType>::LSS() {
    _tol = 1e-6;
    _max_iter = 1000;
}
template<typename MatrixType, typename VectorType>
void LSS<MatrixType, VectorType>::setTol(double tol) {
    _tol = tol;
}

template<typename MatrixType, typename VectorType>
void LSS<MatrixType, VectorType>::setMaxIter(int iter){
    _max_iter = iter;
}

template<typename MatrixType, typename VectorType>
double LSS<MatrixType, VectorType>::getTol() {
    return _tol;
}

template<typename MatrixType, typename VectorType>
int LSS<MatrixType, VectorType>::getMaxIter(){
    return _max_iter;
}


// ================= GENERATION DES MATRICES ==================
pair<SparseMatrix<double>, MatrixXd> generate_simple_sparse_tridiagonal_matrix(int n, double diagonal_value, double off_diagonal_value) {
    vector<Triplet<double>> triplets;

    for(int i=0; i<n; i++) {
        triplets.emplace_back(i, i, diagonal_value);
        if(i < n-1) {
            triplets.emplace_back(i, i+1, off_diagonal_value);
            triplets.emplace_back(i+1, i, off_diagonal_value);
        }
    }

    SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    MatrixXd A_dense = MatrixXd::Zero(n, n);
    for(int i=0; i<n; i++) {
        A_dense(i, i) = diagonal_value;
        if(i < n-1) {
            A_dense(i, i+1) = off_diagonal_value;
            A_dense(i+1, i) = off_diagonal_value;
        }
    }

    return {A, A_dense};
}

// ================== JACOBI DENSE ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>> jacobi_dense_with_error(const MatrixXd& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k< this->getMaxIter(); k++) {
        VectorXd x_new = VectorXd::Zero(n);
        for(int i=0; i<n; i++) {
            double sum_ax = A.row(i).dot(x) - A(i,i)*x(i);
            x_new(i) = (b(i) - sum_ax)/A(i,i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < this->getTol()) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== JACOBI SPARSE ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>>jacobi_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    VectorXd diag = A.diagonal();
    VectorXd diag_inv = diag.cwiseInverse();
    SparseMatrix<double> M = A;
    for(int i=0; i<n; i++) M.coeffRef(i,i) = 0.0;

    for(int k=0; k<this->getMaxIter(); k++) {
        VectorXd x_new = diag_inv.asDiagonal() * (b - M * x);
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < this->getTol()) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= GAUSS-SEIDEL ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>> gauss_seidel_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k<this->getMaxIter(); k++) {
        VectorXd x_new = x;
        for(int i=0; i<n; i++) {
            double s1 = 0.0, s2 = 0.0;
            for(SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if(it.col() < i) s1 += it.value() * x_new(it.col());
                else if(it.col() > i) s2 += it.value() * x(it.col());
            }
            x_new(i) = (b(i) - s1 - s2)/A.coeff(i,i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        if(error < this->getTol()) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= SOR ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>> SOR_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double omega) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k<this->getMaxIter(); k++) {
        VectorXd x_new = x;
        for(int i=0; i<n; i++) {
            double s1 = 0.0, s2 = 0.0;
            for(SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if(it.col() < i) s1 += it.value() * x_new(it.col());
                else if(it.col() > i) s2 += it.value() * x(it.col());
            }
            double s = (b(i) - s1 - s2)/A.coeff(i,i);
            x_new(i) = omega * s + (1-omega) * x(i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        if(error < this->getTol()) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== RAYON SPECTRAL ==================
template<typename MatrixType, typename VectorType>
double LSS<typename MatrixType, typename VectorType>::rayon_spectral_JS(const MatrixType& A) {
    MatrixXd D = A.diagonal().asDiagonal();
    MatrixXd D_inv = D.inverse();
    MatrixXd T = D_inv * (A - D);
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}
template<typename MatrixType, typename VectorType>
double LSS<typename MatrixType, typename VectorType>::rayon_spectral_GS(const MatrixType& A) {
    MatrixXd D = A.diagonal().asDiagonal();
    MatrixXd L = MatrixXd(A.triangularView<Lower>()) - D;
    MatrixXd U = MatrixXd(A.triangularView<Upper>()) - D;
    MatrixXd C = D - L;
    MatrixXd C_inv = C.inverse();
    MatrixXd T = C_inv * U;
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

template<typename MatrixType, typename VectorType>
double LSS<typename MatrixType, typename VectorType>::rayon_spectral_SOR(const MatrixType& A, double omega) {
    MatrixXd D = A.diagonal().asDiagonal();
    MatrixXd L = MatrixXd(A.triangularView<Lower>()) - D;
    MatrixXd U = MatrixXd(A.triangularView<Upper>()) - D;
    MatrixXd M = D - omega*L;
    MatrixXd N = (1-omega)*D + U;
    MatrixXd M_inv = M.inverse();
    MatrixXd T = omega * (M_inv * N);
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

// ================== DIAGONALE DOMINANTE ==================
template<typename MatrixType, typename VectorType>
void LSS<typename MatrixType, typename VectorType>::diagonale_dominante(const MatrixType& A) {
    int n = A.rows();
    for(int i=0; i<n; i++) {
        double somme = 0.0;
        for(int j=0; j<n; j++) {
            if(i != j) somme += abs(A(i,j));
        }
        if(abs(A(i,i)) < somme) {
            cout << "La matrice n'est pas diagonale dominante.\n";
            return;
        }
    }
    cout << "La matrice est diagonale dominante.\n";
}


