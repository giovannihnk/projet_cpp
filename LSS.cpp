#include "LSS.hh"

using namespace Eigen;
using namespace std;

template<typename MatrixType, typename VectorType>
LSS<MatrixType, VectorType>::LSS() : _tol(1e-6), _max_iter(1000) { }

template<typename MatrixType, typename VectorType>
void LSS<MatrixType, VectorType>::setTolerance(double tol) { _tol = tol; }

template<typename MatrixType, typename VectorType>
void LSS<MatrixType, VectorType>::setMaxIterations(int iter) { _max_iter = iter; }

// ================= GENERATION DES MATRICES ==================
template<typename MatrixType, typename VectorType>
pair<SparseMatrix<double>, MatrixType>
LSS<MatrixType, VectorType>::generate_simple_sparse_tridiagonal_matrix(int n, double diagonal_value, double off_diagonal_value) {
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

    MatrixType A_dense = MatrixType::Zero(n, n);
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
tuple<VectorType, int, double, vector<double>>
LSS<MatrixType, VectorType>::jacobi_dense_with_error(const MatrixType& A, const VectorType& b,
                                                     const VectorType& x0, const VectorType& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorType x = x0;
    vector<double> errors;

    for(int k=0; k<_max_iter; k++) {
        VectorType x_new = VectorType::Zero(n);
        for(int i=0; i<n; i++) {
            double sum_ax = A.row(i).dot(x) - A(i,i)*x(i);
            x_new(i) = (b(i) - sum_ax)/A(i,i);
        }
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < _tol) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== JACOBI SPARSE ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>>
LSS<MatrixType, VectorType>::jacobi_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b,
                                                      const VectorType& x0, const VectorType& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorType x = x0;
    vector<double> errors;

    VectorType diag = A.diagonal();
    VectorType diag_inv = diag.cwiseInverse();
    SparseMatrix<double> M = A;
    for(int i=0; i<n; i++) M.coeffRef(i,i) = 0.0;

    for(int k=0; k<_max_iter; k++) {
        VectorType x_new = diag_inv.asDiagonal() * (b - M * x);
        double error = (x_new - x_exact).norm();
        errors.push_back(error);
        x = x_new;
        if(error < _tol) break;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= GAUSS-SEIDEL ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>>
LSS<MatrixType, VectorType>::gauss_seidel_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b,
                                                            const VectorType& x0, const VectorType& x_exact) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorType x = x0;
    vector<double> errors;

    for(int k=0; k<_max_iter; k++) {
        VectorType x_new = x;
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
        if(error < _tol) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================= SOR ==================
template<typename MatrixType, typename VectorType>
tuple<VectorType, int, double, vector<double>>
LSS<MatrixType, VectorType>::SOR_sparse_with_error(const SparseMatrix<double>& A, const VectorType& b,
                                                   const VectorType& x0, const VectorType& x_exact, double omega) {
    auto start = chrono::high_resolution_clock::now();
    int n = A.rows();
    VectorType x = x0;
    vector<double> errors;

    for(int k=0; k<_max_iter; k++) {
        VectorType x_new = x;
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
        if(error < _tol) break;
        x = x_new;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();
    return {x, (int)errors.size(), time_taken, errors};
}

// ================== RAYON SPECTRAL ==================
template<typename MatrixType, typename VectorType>
double LSS<MatrixType, VectorType>::rayon_spectral_JS(const MatrixType& A) {
    MatrixType D = A.diagonal().asDiagonal();
    MatrixType D_inv = D.inverse();
    MatrixType T = D_inv * (A - D);
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

template<typename MatrixType, typename VectorType>
double LSS<MatrixType, VectorType>::rayon_spectral_GS(const MatrixType& A) {
    MatrixType D = A.diagonal().asDiagonal();
    MatrixType L = A.template triangularView<Eigen::Lower>().toDenseMatrix() - D;
    MatrixType U = A.template triangularView<Eigen::Upper>().toDenseMatrix() - D;

    MatrixType C = D - L;
    MatrixType C_inv = C.inverse();
    MatrixType T = C_inv * U;
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

template<typename MatrixType, typename VectorType>
double LSS<MatrixType, VectorType>::rayon_spectral_SOR(const MatrixType& A, double omega) {
   MatrixType D = A.diagonal().asDiagonal();
    MatrixType L = A.template triangularView<Eigen::Lower>().toDenseMatrix() - D;
    MatrixType U = A.template triangularView<Eigen::Upper>().toDenseMatrix() - D;
    MatrixType M = D - omega*L;
    MatrixType N = (1-omega)*D + U;
    MatrixType M_inv = M.inverse();
    MatrixType T = omega * (M_inv * N);
    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for(int i=0; i<eigvals.size(); i++) rho = max(rho, abs(eigvals(i)));
    return rho;
}

// ================== DIAGONALE DOMINANTE ==================
template<typename MatrixType, typename VectorType>
void LSS<MatrixType, VectorType>::diagonale_dominante(const MatrixType& A) {
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

// ================== INSTANTIATION EXPLICITE ==================
template class LSS<MatrixXd, VectorXd>;
