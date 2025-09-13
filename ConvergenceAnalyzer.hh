#ifndef __CONVERGENCEANALYZER_HH__
#define __CONVERGENCEANALYZER_HH__

#include "eigen/Eigen/Dense"
#include <iostream>

class ConvergenceAnalyzer {
public:
    // Calcule le rayon spectral de la matrice d’itération Jacobi
    static double spectralRadiusJacobi(const Eigen::MatrixXd& A);

    // Calcule le rayon spectral de la matrice d’itération Gauss-Seidel
    static double spectralRadiusGaussSeidel(const Eigen::MatrixXd& A);

    // Calcule le rayon spectral de la matrice d’itération SOR (Successive Over-Relaxation)
    static double spectralRadiusSOR(const Eigen::MatrixXd& A, double omega);

    // Vérifie si la matrice est diagonale dominante
    static void checkDiagonalDominance(const Eigen::MatrixXd& A);
};

#endif
