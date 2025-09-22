Ce dépôt github reprend un de nos projets portant sur la résolution de système linéaire à grande dimension, au cours de ce projet nous avons pu implémenter en Python les méthodes de :
- Jacobi (version dense)
- Jacobi (version sparse)
- Gauss-Seidel (version dense)
- Gauss-Seidel (version sparse)
- SOR (version dense)
- SOR (version sparse)

Nous avons pu retranscrire ce projet du point de vu de la POO (Programmation Orientée Objet). 
Le premier pas fut de choisir une architecture de projet adapté, on entend par cela les choix d'encapsulage, les liens entres nos différentes classes et leur natures.
Le but était d'optimiser notre code via le polymorphisme et l'héritage de C++, permettant ainsi d'éviter les redondances et les possibles erreurs dûes à des modifications locales (au sein d'une classe et pas les autres). 
