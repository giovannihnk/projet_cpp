This GitHub repository contains one of our projects on large-scale linear system solving (SLGD): Ax = b. During this project, we implemented the following methods in Python:
• Jacobi (dense version)
• Jacobi (sparse version)
• Gauss-Seidel (dense version)
• Gauss-Seidel (sparse version)
• SOR (dense version)
• SOR (sparse version)
We were able to transcribe this project from an OOP (Object-Oriented Programming) perspective. The first step was to choose a suitable project architecture, by which we mean the choices of encapsulation, the links between our different classes and their nature. The goal was to optimise our code using C++ polymorphism and inheritance, thereby avoiding redundancies and potential errors due to local modifications (within one class and not others). Another major advantage was C++'s native memory management, which ultimately allowed us to compare our implementations (Python vs C++).
The conclusion is clear: Python is faster when the matrix 
𝐴
 in question is small. Conversely, when the dimension of A becomes too large, the opposite is true.
