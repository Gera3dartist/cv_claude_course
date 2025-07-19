# Eigenvalues and Eigenvectors - 6 Practical Assignments

## Assignment 1: Foundation - Computing Eigenvalues by Hand
**Format:** Paper + Jupyter verification

### Paper Work:
Find eigenvalues and eigenvectors for these matrices using the characteristic equation det(A - λI) = 0:

1. **2×2 Matrix:**
   ```
   A₁ = [3  1]
        [0  2]
   ```

2. **2×2 Symmetric Matrix:**
   ```
   A₂ = [4  2]
        [2  1]
   ```

3. **3×3 Upper Triangular:**
   ```
   A₃ = [2  1  3]
        [0  1  2]
        [0  0  4]
   ```

### Jupyter Component:
- Verify your hand calculations using `numpy.linalg.eig()`
- Plot the eigenvectors for the 2×2 matrices
- Compute condition numbers and discuss numerical stability

**Learning Goal:** Master the fundamental computational process and develop number sense for eigenvalue problems.

---

## Assignment 2: Geometric Intuition - Linear Transformations
**Format:** Paper analysis + Jupyter visualization

### Paper Work:
For each transformation matrix, predict what happens to the unit circle and standard basis vectors:

1. **Scaling Matrix:**
   ```
   S = [3  0]
       [0  0.5]
   ```

2. **Rotation + Scaling:**
   ```
   R = [0  -2]
       [2   0]
   ```

3. **Reflection Matrix:**
   ```
   F = [1   0]
       [0  -1]
   ```

### Jupyter Component:
- Create animated visualizations showing how these matrices transform:
  - A grid of points
  - The unit circle
  - Random vectors
- Plot eigenvectors as special directions that don't change direction
- Compare your predictions with the actual transformations

**Learning Goal:** Build geometric intuition for how eigenvalues/eigenvectors relate to linear transformations.

---

## Assignment 3: Diagonalization Deep Dive
**Format:** Paper + Jupyter implementation

### Paper Work:
1. **Diagonalize by hand:**
   ```
   A = [1  2]
       [2  1]
   ```
   Find P and D such that A = PDP⁻¹

2. **Non-diagonalizable case:**
   ```
   B = [1  1]
       [0  1]
   ```
   Explain why this matrix cannot be diagonalized.

3. **Powers of matrices:** Use diagonalization to compute A¹⁰ for matrix A above.

### Jupyter Component:
- Implement your own diagonalization function (don't use built-in)
- Create a function to compute matrix powers using diagonalization
- Compare computational efficiency: A^n vs diagonalization method for large n
- Visualize the effect of repeated applications of the transformation

**Learning Goal:** Understand diagonalization as a change of basis and its computational advantages.

---

## Assignment 4: Real-World Application - Vibrating Systems
**Format:** Paper modeling + Jupyter simulation

### Paper Work:
Model a 2-mass spring system:
```
m₁ ↔ spring₁ ↔ m₂ ↔ spring₂ ↔ wall
```

1. **Derive the system:** Set up the differential equation d²x/dt² = -Kx where K is the stiffness matrix
2. **Find normal modes:** The eigenvalues of K give you the natural frequencies
3. **Physical interpretation:** What do the eigenvectors represent physically?

### Jupyter Component:
- Simulate the system with different initial conditions
- Animate the motion showing how each normal mode oscillates
- Create a function that decomposes any initial condition into normal modes
- Plot frequency response and identify resonance frequencies

**Learning Goal:** Connect mathematical concepts to physical systems and understand modal analysis.

---

## Assignment 5: PCA Implementation from Scratch
**Format:** Paper theory + Jupyter implementation

### Paper Work:
1. **Derive PCA:** Starting from the covariance matrix, show why we need its eigenvectors
2. **Dimensionality reduction:** Explain how eigenvalues determine information content
3. **Centering data:** Why do we subtract the mean before computing covariance?

### Jupyter Component:
- Generate 2D correlated data (e.g., height vs weight simulation)
- Implement PCA from scratch:
  ```python
  def my_pca(data, n_components):
      # Your implementation here
      pass
  ```
- Compare with sklearn's PCA
- Visualize:
  - Original data with principal components overlaid
  - Data in the new coordinate system
  - Reconstruction error vs number of components
- Apply to a real dataset (e.g., iris or handwritten digits)

**Learning Goal:** Master the most important application of eigendecomposition in data science.

---

## Assignment 6: Advanced Topics - Spectral Analysis
**Format:** Paper + Jupyter exploration

### Paper Work:
1. **Spectral theorem:** State and explain why symmetric matrices have special properties
2. **Positive definite matrices:** How do eigenvalues determine positive definiteness?
3. **Condition number:** Relate eigenvalues to numerical stability

### Jupyter Component:
- **Graph Laplacian:** Create a simple graph, compute its Laplacian matrix, and analyze eigenvalues
  - Connect eigenvalues to graph connectivity
  - Use the second smallest eigenvalue (Fiedler value) for graph partitioning
- **Markov chains:** Create a transition matrix, find steady state using eigenvectors
- **Google PageRank simulation:** Implement a simplified PageRank algorithm
- **Stability analysis:** Analyze a linear dynamical system dx/dt = Ax
  - Determine stability from eigenvalue real parts
  - Visualize phase portraits

**Learning Goal:** See eigenvalues as a fundamental tool across mathematics, connecting linear algebra to graph theory, probability, and dynamical systems.

---

## Grading Criteria:
- **Computational accuracy** (30%): Correct eigenvalue/eigenvector calculations
- **Geometric understanding** (25%): Clear explanations of what eigenvalues/eigenvectors represent
- **Code quality** (20%): Clean, well-commented implementations
- **Visualizations** (15%): Clear, informative plots that support understanding
- **Connections** (10%): Ability to relate concepts across different contexts

## Recommended Timeline:
- **Day 1:** Assignments 1-2 (Build foundation and intuition)
- **Day 2:** Assignments 3-4 (Deepen understanding with applications)
- **Day 3:** Assignments 5-6 (Advanced applications and connections)

## Additional Resources:
- 3Blue1Brown's "Essence of Linear Algebra" videos on eigenvalues
- Interactive eigenvalue visualization tools
- Papers on spectral graph theory for Assignment 6