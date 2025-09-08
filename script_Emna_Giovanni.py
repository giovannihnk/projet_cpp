import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt
import time


########## GENEARTION DES MATRICES ################
def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        A_dense: equivalent Dense matrix (numpy array)
        b: Right-hand side vector (numpy array).
    """
    main_diag = np.full(n, diagonal_value)  #on recupere toutes les valeurs de la diagonale
    off_diag1 = np.full((n-1), off_diagonal_value)   #et celles sur la sur et sous diagonale
    off_diag2 = np.full((n-1), off_diagonal_value)
    #print(main_diag,off_diag1,off_diag2)

    # Constructtion de la matrice creuse
    data = np.concatenate((main_diag,off_diag1,off_diag2))
    #print(data)
    #construction des coordonnées des lignes et colonnes
    rc1 = np.arange(n) 
    rc2 = np.arange(n-1)
    rc3 = np.arange(1,n)
    rows = np.concatenate((rc1,rc2,rc3))
   # print(rows)
    cols = np.concatenate((rc1,rc3,rc2))
    #print(cols)

    As = csr_matrix((data, (rows, cols)), shape=(n, n))
    #print(As)

    # Construction de la matrice dense
    A_dense = np.zeros((n, n))
    for i in range(n):
        A_dense[i, i] = diagonal_value
    A_dense = A_dense + np.diag(off_diag1,k=1) + np.diag(off_diag2,k=-1)
    b =  np.random.rand(n)
    #print(A_dense)
    return As, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    return generate_simple_sparse_tridiagonal_matrix(n,2,-1)


############ IMPLEMENTATION DES METHODES ###############

def jacobi_dense_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=1000):
    """
    Jacobi method for dense matrices.

    Args:
        A: Dense coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    start_time = time.time()  #calcul du temps d'execution
    n = A.shape[0]  #taille de la matrice
    x = x0.copy()   #solution initiale
    errors = []   #liste pour stocker les erreurs à chaque itérations
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            # Calcul de la somme des termes hors-diagonaux
            sum_ax = np.dot(A[i,:],x)-A[i,i]*x[i]
            # Mise à jour de x_new[i]
            x_new[i] = (b[i]-sum_ax)/A[i,i]

        error = np.linalg.norm(x_new - x_exact)   #calcul de l'erreur
        errors.append(error)
        x = x_new
        if error < tol:   #verification de la convergence par rapport à la tolérence  
            break

    end_time = time.time()
    time_taken = end_time - start_time
    return x, k+1, time_taken, errors  #retourne la solution finale, le nb d'itértions, le temps d'execution et les erreurs

def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    """
    Jacobi method for sparse matrices.

    Args:
        A: Sparse coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess for the solution vector.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        k: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors : list of errors for each iterations
    """
    start_time = time.time()
    n = A.shape[0]
    x = x0.copy()
    errors = []
    diag = [1/v if v!=0 else 0 for v in A.diagonal()]   #inversion de la matrice diagonale
    diag_csr = diags(A.diagonal(), offsets=0, format='csr') #matrice diagonale sous format csr
    diag_inv = diags(diag, offsets=0, format='csr')   #matrice diagonale inverse sous forme csr
    M = A-diag_csr
    for k in range(max_iter):
        x_new = diag_inv @ (b - M @ x)
        error = np.linalg.norm(x_new - x_exact)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    return x, k+1, time_taken, errors


def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=1e-6, max_iter=10000):
    """
    Gauss Seidel method for sparse matrices.

    Args:
        A: Sparse coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess for the solution vector.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        k: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors : list of errors for each iterations
    """
    start_time = time.time()
    n = A.shape[0]
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            #calcul des sommes à droite et à gauche de la diagonale
            s1 = A[i, :i].dot(x_new[:i]).item()
            s2 = A[i, i+1:].dot(x_new[i+1:]).item()
            x_new[i] = (b[i] - s1 - s2)/A[i,i] #mise à jour de x_new à l'aide de la formule de gausse seidel
        error = np.linalg.norm(x_new-x_exact) 
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time = time.time()
    time_taken = end_time - start_time

    return x, k+1, time_taken, errors


def SOR_sparse_with_error(A,b,x0,x_exact,omega,tol=1e-6,max_iter=10000):
    """
    SOR method for sparse matrices.

    Args:
        A: Sparse coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess for the solution vector.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        k: Number of iterations performed.
        time_taken: Time taken for the iterations.
        errors : list of errors for each iterations
    """
    start_time = time.time()
    n = A.shape[0]
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        x_new = x.copy()
        s = x.copy()
        for i in range(n):
            #calcul des sommes à droite et à gauche de la diagonale
            s1 = A[i, :i].dot(x_new[:i]).item()
            s2 = A[i, i+1:].dot(x_new[i+1:]).item()
            #calcul de la solution intermédiaire avec la formule de gauss seidel
            s[i] = (b[i] - s1 - s2)/A[i,i]
            x_new[i] = omega * s[i] + (1-omega) * x[i]  #mise à jour de x_new avec la formule SOR
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time = time.time()
    time_taken = end_time - start_time

    return x, k+1, time_taken, errors



#### RAYON SPECTRAL POUR MATRICE DENSE ####

def rayon_spectral_JS(A):
    ###calcul du rayon spectral pour Jacobi dense
    D = np.diag(np.diag(A))
    #print("MATRICE D : ",D)
    D_inv = np.linalg.inv(D)
    T = D_inv @ (A-D)  
    val_propre = np.linalg.eigvals(T)
    ray_spectral = max(abs(val_propre))
    #print("le rayon spectral est : ",ray_spectral)
    return ray_spectral


def rayon_spectral_GS (A):
   #Calcul du rayon spectral pour GS
   D = np.diag(np.diag(A))
   L = np.tril(A,-1)
   U = np.triu(A,1)
   C = D - L
   C_inv = np.linalg.inv(C)
   #print("MATRICE INV : ",C_inv)
   T = C_inv @ U  
   val_propre = np.linalg.eigvals(T)
   ray_spectral = max(abs(val_propre))
   #print("le rayon spectral est : ",ray_spectral)
   return ray_spectral

def rayon_spectral_SOR (A,omega):
   #Calcul du rayon spectral pour SOR
   D = np.diag(np.diag(A))
   # Préconditionneur de SOR
   L = np.tril(A,-1)
   U = np.triu(A,1)
   M = D - L*omega
   N = D*(1  - omega) + U
   M_inv = np.linalg.inv(M)
   #print("MATRICE INV : ",M_inv)
   T = M_inv @ N * omega
   val_propre = np.linalg.eigvals(T)
   ray_spectral = max(abs(val_propre))
   #print("le rayon spectral est : ",ray_spectral)
   return ray_spectral


#### DIAGONALE DOMINANTE JACOBI DENSE ####
def diagonale_dominante(A):
# Cette fonction détermine la nature diagonale dominante d'une matrice dense.
    n = A.shape[0]
    for i in range(n):
        # Calcul de la somme des éléments non-diagonaux
        somme_hors_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        if (abs(A[i, i]) < somme_hors_diag):
            print("La matrice n'est pas diagonale dominante.")
            return 
    print("La matrice est diagonale dominante.")
    return


####### PLOT (echelle log sur l'axe y) #######

def plot_taille_temps(taille, temps_JS, temps_GS, temps_JD):
    """
    Affiche un graphique avec 3 courbes:
        Jacobi dense
        Jacobi Sparse
        Gauss Seidel
    Le temps d'execution de chaque méthode en fonction de la taille de la matrice
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(taille, temps_JS, label = "Jacobi Sparse", marker='o', linestyle='-',color="blue")
    plt.semilogy(taille, temps_GS, label = "Gauss Seidel", marker='o', linestyle='-',color="green")
    plt.semilogy(taille, temps_JD, label = "Jacobi Dense", marker='o', linestyle='-',color="red")  
    plt.title("temps en fonction de la taille")
    plt.xlabel("taille n")
    plt.ylabel("temps")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grille adaptée aux log
    plt.show()



def plot_SOR_GS(taille, temps_GS,temps_SOR, omega):
    """
    Affiche un graphique avec temps d'execution en fonction de la taille de la matrice pour:
        SOR pour tout omega compris entre 0.1 et 1.9 (avec un pas de 0.1)
        Gauss Seidel
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(taille, temps_GS, label = "Gauss Seidel", marker='o', linestyle='-',color="orange")  
    num_omega = len(omega)
    for i in range(num_omega):
        temps_omega = [temps_SOR[j * num_omega + i] for j in range(len(taille))]
        plt.semilogy(taille, temps_omega, label = f"SOR : omega = {omega[i]}", marker='o', linestyle='-') 
    plt.title("temps en fonction de la taille")
    plt.xlabel("taille n")
    plt.ylabel("temps")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grille adaptée aux log
    plt.show()


def plot_iter_errors_SOR(iteration,errors_SOR,val_omega):
    """
    Affiche un graphique avec les erreurs en fonction du nombre d'itérations pour:
        SOR pour tout omega compris entre 0.1 et 1.9 (avec un pas de 0.1)
    """
    plt.figure(figsize=(12, 8))
    len_omega=len(val_omega)
    #colors = get_cmap("viridis")

    for i in range(len_omega):
        # Tracer les erreurs pour chaque valeur de omega
        plt.semilogy(iteration[i], errors_SOR[i], label=f"omega = {val_omega[i]:.1f}")#, color=colors(i / len_omega))
    
    plt.xlabel("Iterations")
    plt.ylabel("Error ")
    plt.title("Error vs Iterations pour diférentes valeurs d'omega")
    plt.grid(True)#,which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def plot_iter_error(iter_JS,iter_GS,iter_JD, error_JS, error_GS, error_JD):
    """
    Affiche un graphique avec 3 courbes :
        Jacobi dense
        Jacobi Sparse
        Gauss Seidel
    Les erreurs en fonction du nombres d'itérations pour ces 3 méthodes
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(iter_JS), error_JS, label = "Jacobi Sparse", marker='o', linestyle='-')
    plt.semilogy(range(iter_GS), error_GS, label = "Gauss Seidel", marker='o', linestyle='-')
    plt.semilogy(range(iter_JD), error_JD, label = "Jacobi Dense", marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.title("erreur vs iterations ")
    plt.xlabel("itérations")
    plt.ylabel("erreurs")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grille adaptée aux log
    plt.show()




### EXEMPLE
    
taille = [10,20,30]
times_JS = []
times_GS = []
times_JD = []
times_SOR = []
all_errors_SOR = []
all_iterations_SOR = []
O = [i/10 for i in range(1,20)]
for n in taille:
    x0 = np.zeros(n)  ## initial guess
    A_sparse, A_dense, b = generate_sparse_tridiagonal_matrix(n)
    r1 = rayon_spectral_JS(A_dense)
    r2 = rayon_spectral_GS(A_dense)
    r3 = rayon_spectral_SOR(A_dense,1.8)
    print (f"rayon Jacobi pour {n} : {r1}")
    print (f"rayon GS pour {n} : {r2}")
    print (f"rayon SOR pour {n} : {r3}")
    b[0] = 0
    b[n-1] = 0
    h=1/(n+1)
    b_h = (h**2)*b
    x_exact = np.linalg.solve(A_dense, b_h)
    # print("Exact Solution : ", x_exact)
    xJD, iterJD, time_takenJd, errorsJD = jacobi_dense_with_error(A_dense,b_h,x0,x_exact)
    xJS, iterJS, time_takenJS, errorsJS = jacobi_sparse_with_error(A_sparse,b_h,x0,x_exact)
    xGS, iterGS, time_takenGS, errorsGS = gauss_seidel_sparse_with_error(A_sparse,b_h,x0,x_exact)
    
    times_JS.append(time_takenJS)
    times_GS.append(time_takenGS)
    times_JD.append(time_takenJd)
    errors_SOR_for_omega = []
    iterations_for_omega = []
    for omega in O:
        xSOR, iterSOR, time_takenSOR, errorsSOR = SOR_sparse_with_error(A_sparse,b_h,x0,x_exact,omega)
        times_SOR.append(time_takenSOR)
        errors_SOR_for_omega.append(errorsSOR)
        iterations_for_omega.append(range(len(errorsSOR)))  # Collecter les itérations correspondantes

    all_errors_SOR.append(errors_SOR_for_omega)
    all_iterations_SOR.append(iterations_for_omega)

#affichage des plot
plot_iter_errors_SOR(all_iterations_SOR[0], all_errors_SOR[0],O)
plot_taille_temps(taille,times_JS,times_GS,times_JD)
plot_SOR_GS(taille,times_GS,times_SOR,O)
plot_iter_error(iterJS,iterGS,iterJD, errorsJS, errorsGS, errorsJD)

