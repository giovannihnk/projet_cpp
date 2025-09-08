#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Sparse"
#include <chrono>
#include <cmath>

using namespace Eigen;
using namespace std;

// ================= GENERATION DES MATRICES ==================
pair<SparseMatrix<double>, MatrixXd> generate_simple_sparse_tridiagonal_matrix(int n, double diagonal_value=10, double off_diagonal_value=4) {
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

pair<SparseMatrix<double>, MatrixXd> generate_sparse_tridiagonal_matrix(int n) {
    return generate_simple_sparse_tridiagonal_matrix(n, 2, -1);
}

// ================== JACOBI DENSE ==================
tuple<VectorXd, int, double, vector<double>> jacobi_dense_with_error(const MatrixXd& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double tol=1e-6, int max_iter=1000) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k<max_iter; k++) {
        VectorXd x_new = VectorXd::Zero(n);
        for(int i=0; i<n; i++) {
            double sum_ax = A.row(i).dot(x) - A(i,i)*x(i);
            x_new(i) = (b(i) - sum_ax)/A(i,i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < tol) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== JACOBI SPARSE ==================
tuple<VectorXd, int, double, vector<double>> jacobi_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double tol=1e-6, int max_iter=10000) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    VectorXd diag = A.diagonal();
    VectorXd diag_inv = diag.cwiseInverse();
    SparseMatrix<double> M = A;
    for(int i=0; i<n; i++) M.coeffRef(i,i) = 0.0;

    for(int k=0; k<max_iter; k++) {
        VectorXd x_new = diag_inv.asDiagonal() * (b - M * x);
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < tol) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= GAUSS-SEIDEL ==================
tuple<VectorXd, int, double, vector<double>> gauss_seidel_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double tol=1e-6, int max_iter=10000) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k<max_iter; k++) {
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
        if(error < tol) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= SOR ==================
tuple<VectorXd, int, double, vector<double>> SOR_sparse_with_error(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const VectorXd& x_exact, double omega, double tol=1e-6, int max_iter=10000) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorXd x = x0;
    vector<double> errors;

    for(int k=0; k<max_iter; k++) {
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
        if(error < tol) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== RAYON SPECTRAL ==================
double rayon_spectral_JS(const MatrixXd& A) {
    MatrixXd D = A.diagonal().asDiagonal();
    MatrixXd D_inv = D.inverse();
    MatrixXd T = D_inv * (A - D);
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

double rayon_spectral_GS(const MatrixXd& A) {
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

double rayon_spectral_SOR(const MatrixXd& A, double omega) {
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
void diagonale_dominante(const MatrixXd& A) {
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

// ================== MAIN ==================
int main() {
    vector<int> tailles = {10,20,30};
    for(int n : tailles) {
        VectorXd x0 = VectorXd::Zero(n);
        auto [A_sparse, A_dense] = generate_sparse_tridiagonal_matrix(n);

        double r1 = rayon_spectral_JS(A_dense);
        double r2 = rayon_spectral_GS(A_dense);
        double r3 = rayon_spectral_SOR(A_dense, 1.8);

        cout << "Rayon Jacobi pour " << n << " : " << r1 << endl;
        cout << "Rayon GS pour " << n << " : " << r2 << endl;
        cout << "Rayon SOR pour " << n << " : " << r3 << endl;

        VectorXd b = VectorXd::Random(n);
        b(0) = 0; b(n-1) = 0;
        double h = 1.0/(n+1);
        VectorXd b_h = h*h*b;

        VectorXd x_exact = A_dense.colPivHouseholderQr().solve(b_h);

        auto [xJD, iterJD, timeJD, errorsJD] = jacobi_dense_with_error(A_dense,b_h,x0,x_exact);
        auto [xJS, iterJS, timeJS, errorsJS] = jacobi_sparse_with_error(A_sparse,b_h,x0,x_exact);
        auto [xGS, iterGS, timeGS, errorsGS] = gauss_seidel_sparse_with_error(A_sparse,b_h,x0,x_exact);

        cout << "Jacobi Dense itérations: " << iterJD << " temps: " << timeJD << "s\n";
        cout << "Jacobi Sparse itérations: " << iterJS << " temps: " << timeJS << "s\n";
        cout << "Gauss-Seidel itérations: " << iterGS << " temps: " << timeGS << "s\n";
    }
    return 0;
}
