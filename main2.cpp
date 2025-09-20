#include <iostream>
#include <memory>
#include "MatrixGenerator.hh"
#include "JacobiSolver.hh"
#include "GaussSeidelSolver.hh"
#include "SORSolver.hh"
#include "ConvergenceAnalyzer.hh"   
#include <iomanip> 


using namespace std;
using namespace Eigen;

int main() {
    int n = 110;
    auto [A_sparse, A_dense] = MatrixGenerator::generateTridiagonal(n, 4.0, -1.0);

    VectorXd x_exact = VectorXd::Ones(n);
    VectorXd b = A_dense * x_exact;
    VectorXd x0 = VectorXd::Zero(n);

    // 🔹 Analyse de convergence avant de lancer les solveurs
    cout << "=== Analyse de convergence ===" << endl;
    ConvergenceAnalyzer::checkDiagonalDominance(A_dense);
    cout << "Rayon spectral (Jacobi)       : " << std::fixed << std::setprecision(16) 
         << ConvergenceAnalyzer::spectralRadiusJacobi(A_dense) << endl;
    cout << "Rayon spectral (Gauss-Seidel) : " << std::fixed << std::setprecision(16) 
         << ConvergenceAnalyzer::spectralRadiusGaussSeidel(A_dense) << endl;
    cout << "Rayon spectral (SOR, omega=1.8)   : " << std::fixed << std::setprecision(16) 
         << ConvergenceAnalyzer::spectralRadiusSOR(A_dense, 1.8) << endl;
    cout << "==============================" << endl << endl;

    unique_ptr<IterativeSolver> solver;

    // Jacobi
    solver = make_unique<JacobiSolver>(1e-6, 1000);
    SolveResult resJ = solver->solve(A_sparse, b, x0, x_exact);
    cout << "Jacobi: " << resJ.iterations 
         << " iterations, Final error = " << resJ.errors.back()
         << ", time = " << resJ.time << endl;

    // Gauss-Seidel
    solver = make_unique<GaussSeidelSolver>(1e-6, 1000);
    SolveResult resGS = solver->solve(A_sparse, b, x0, x_exact);
    cout << "Gauss-Seidel: " << resGS.iterations 
         << " iterations, Final error = " << resGS.errors.back() 
         << ", time = " << resGS.time << endl;

    // SOR
    solver = make_unique<SORSolver>(1.8, 1e-6, 1000);
    SolveResult resSOR = solver->solve(A_sparse, b, x0, x_exact);
    cout << "SOR (omega=1.8): " << resSOR.iterations 
         << " iterations, Final error = " << resSOR.errors.back()
         << ", time = " << resSOR.time << endl;

    return 0;
}
