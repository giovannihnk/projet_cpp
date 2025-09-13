#include "ConvergenceAnalyzer.hh"

double ConvergenceAnalyzer::spectralRadiusJacobi(const Eigen::MatrixXd& A) {
    Eigen::MatrixXd D = A.diagonal().asDiagonal();
    Eigen::MatrixXd D_inv = D.inverse();
    Eigen::MatrixXd T = D_inv * (A - D);
    Eigen::VectorXcd eigvals = T.eigenvalues();

    double rho = 0.0;
    for (int i = 0; i < eigvals.size(); i++)
        rho = std::max(rho, std::abs(eigvals(i)));
    return rho;
}

double ConvergenceAnalyzer::spectralRadiusGaussSeidel(const Eigen::MatrixXd& A) {
    Eigen::MatrixXd D = A.diagonal().asDiagonal();
    Eigen::MatrixXd L = A.triangularView<Eigen::Lower>().toDenseMatrix() - D;
    Eigen::MatrixXd U = A.triangularView<Eigen::Upper>().toDenseMatrix() - D;

    Eigen::MatrixXd C = D - L;
    Eigen::MatrixXd C_inv = C.inverse();
    Eigen::MatrixXd T = C_inv * U;

    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for (int i = 0; i < eigvals.size(); i++)
        rho = std::max(rho, std::abs(eigvals(i)));
    return rho;
}

double ConvergenceAnalyzer::spectralRadiusSOR(const Eigen::MatrixXd& A, double omega) {
    Eigen::MatrixXd D = A.diagonal().asDiagonal();
    Eigen::MatrixXd L = A.triangularView<Eigen::Lower>().toDenseMatrix() - D;
    Eigen::MatrixXd U = A.triangularView<Eigen::Upper>().toDenseMatrix() - D;

    Eigen::MatrixXd M = D - omega * L;
    Eigen::MatrixXd N = (1 - omega) * D + U;

    Eigen::MatrixXd M_inv = M.inverse();
    Eigen::MatrixXd T = omega * (M_inv * N);

    Eigen::VectorXcd eigvals = T.eigenvalues();
    double rho = 0.0;
    for (int i = 0; i < eigvals.size(); i++)
        rho = std::max(rho, std::abs(eigvals(i)));
    return rho;
}

void ConvergenceAnalyzer::checkDiagonalDominance(const Eigen::MatrixXd& A) {
    int n = A.rows();
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) sum += std::abs(A(i, j));
        }
        if (std::abs(A(i, i)) < sum) {
            std::cout << "La matrice n'est pas diagonale dominante." << std::endl;
            return;
        }
    }
    std::cout << "La matrice est diagonale dominante." << std::endl;
}
