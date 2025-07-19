"""
Practice: Solve linear systems using NumPy.linalg
# Linear Systems Practice Assignments - Week 1 Day 3-4

"""
import numpy as np


def assignement_warmup():
    """
    ## Assignment 1: Basic 2x2 System (Warmup)
    **Difficulty**: ⭐  
    **Time**: 30 minutes  
    **Goal**: Master the fundamentals

    ### Problem
    Solve this system by hand first, then verify with NumPy:
    ```
    2x + 3y = 7
    4x - y = 1
    y = 4x - 1 -> 2x + 3(4x - 1) = 7 -> 2x + 12x - 3 = 7  ->  14x = 10  -> x = 10/14 = 5/7
    y = 4(5/7) - 7/7 = (20 - 7)/ = 13/7

    ```

    ### Implementation Tasks
    ```python
    import numpy as np

    # 1. Set up the coefficient matrix A and vector b
    # 2. Solve using np.linalg.solve()
    # 3. Verify solution by computing A @ x - b (should be ~0)
    # 4. Check determinant and condition number
    # 5. Visualize the lines and intersection point
    ```

    ### Learning Outcomes
    - Understand matrix-vector representation
    - Learn to verify solutions
    - Introduction to condition numbers

    ---
    """
    A = np.array([[2, 3], [4, -1]])
    b = np.array([7, 1])
    x = np.linalg.solve(A, b)
    # 3. Verify solution by computing A @ x - b (should be ~0)
    A @ x - b  # rougly zero
    # 4. Check determinant and condition number
    det = np.linalg.det(A)
    # 4.1 condition number
    np.linalg.cond(A)





    



"""
## Assignment 2: Overdetermined System with Least Squares
**Difficulty**: ⭐⭐  
**Time**: 45 minutes  
**Goal**: Handle more equations than unknowns

### Problem
Fit a line y = mx + b to these noisy data points:
```
Points: (1, 2.1), (2, 3.9), (3, 6.2), (4, 7.8), (5, 10.1)
```

### Implementation Tasks
```python
# 1. Set up overdetermined system Ax = b
#    where A = [[x₁, 1], [x₂, 1], ...] and x = [m, b]
# 2. Solve using np.linalg.lstsq()
# 3. Compare with np.linalg.solve() and observe the error
# 4. Plot original points and fitted line
# 5. Calculate residual sum of squares
```

### Learning Outcomes
- Understand when exact solutions don't exist
- Learn least squares methodology
- Visualize fitting quality

---

## Assignment 3: Homogeneous System (Null Space)
**Difficulty**: ⭐⭐  
**Time**: 1 hour  
**Goal**: Solve Ax = 0 systems

### Problem
Find non-trivial solutions to this homogeneous system:
```
x + 2y + 3z = 0
2x + 4y + 6z = 0
3x + 6y + 9z = 0
```

### Implementation Tasks
```python
# 1. Recognize this is a rank-deficient system
# 2. Use SVD to find the null space
# 3. Verify that A @ x ≈ 0 for your solution
# 4. Find all solutions (parametric form)
# 5. Visualize the solution space (2D plane in 3D)
```

### Learning Outcomes
- Understand rank deficiency
- Learn SVD for null space computation
- Geometric interpretation of solution spaces

---

## Assignment 4: 2D Point Transformation
**Difficulty**: ⭐⭐  
**Time**: 1 hour  
**Goal**: Apply to basic computer vision

### Problem
You have 4 corners of a square and their transformed positions:
```
Original: (0,0), (1,0), (1,1), (0,1)
Transformed: (1,1), (2,1), (2,2), (1,2)
```

Find the 2x2 transformation matrix + translation vector.

### Implementation Tasks
```python
# 1. Set up system for affine transformation: y = Ax + t
# 2. Separate into rotation/scaling (A) and translation (t)
# 3. Solve for transformation parameters
# 4. Apply transformation to new points
# 5. Visualize original and transformed shapes
```

### Learning Outcomes
- Connect linear algebra to geometric transformations
- Understand affine transformations
- Separate different transformation components

---

## Assignment 5: Simple Camera Model
**Difficulty**: ⭐⭐⭐  
**Time**: 1.5 hours  
**Goal**: Basic perspective projection

### Problem
A simple pinhole camera projects 3D points to 2D:
```
3D points: (1,1,2), (2,1,3), (1,2,4), (2,2,5)
2D projections: (100,100), (133,66), (50,100), (80,80)
```

Find the camera matrix P (3x4) using the equation: s * [u,v,1]ᵀ = P * [X,Y,Z,1]ᵀ

### Implementation Tasks
```python
# 1. Set up the system using Direct Linear Transform (DLT)
# 2. Handle the scale factor s by cross products
# 3. Solve the homogeneous system Ap = 0
# 4. Reshape solution back to 3x4 matrix
# 5. Test projection on new 3D points
```

### Learning Outcomes
- Understand perspective projection mathematics
- Learn Direct Linear Transform method
- Handle homogeneous coordinates

---

## Assignment 6: Homography Estimation
**Difficulty**: ⭐⭐⭐  
**Time**: 1.5 hours  
**Goal**: Image-to-image transformation

### Problem
You have corresponding points between two images of the same planar object:
```
Image 1: (50,50), (150,50), (150,150), (50,150), (100,100)
Image 2: (30,40), (170,30), (180,140), (40,160), (105,90)
```

Find the 3x3 homography matrix H.

### Implementation Tasks
```python
# 1. Set up DLT equations for homography
# 2. Each point pair gives 2 equations
# 3. Solve the homogeneous system
# 4. Reshape to 3x3 matrix and normalize
# 5. Test forward and backward projection
# 6. Visualize point correspondences
```

### Learning Outcomes
- Master homography mathematics
- Understand projective transformations
- Learn point correspondence handling

---

## Assignment 7: Stereo Triangulation
**Difficulty**: ⭐⭐⭐⭐  
**Time**: 2 hours  
**Goal**: 3D reconstruction basics

### Problem
Given two calibrated cameras and corresponding points, triangulate 3D coordinates:
```
Camera 1 matrix P1 = [[500,0,320,0], [0,500,240,0], [0,0,1,0]]
Camera 2 matrix P2 = [[500,0,320,-500], [0,500,240,0], [0,0,1,0]]
Point in image 1: (340, 250)
Point in image 2: (300, 250)
```

### Implementation Tasks
```python
# 1. Set up triangulation equations using cross products
# 2. Form the 4x4 system from both cameras
# 3. Solve using SVD (homogeneous least squares)
# 4. Convert from homogeneous to 3D coordinates
# 5. Verify by projecting back to both images
# 6. Handle numerical precision issues
```

### Learning Outcomes
- Understand stereo geometry
- Learn robust triangulation methods
- Handle homogeneous 3D coordinates

---

## Assignment 8: Bundle Adjustment (Simplified)
**Difficulty**: ⭐⭐⭐⭐  
**Time**: 2.5 hours  
**Goal**: Multi-view optimization

### Problem
You have 3 views of 4 3D points. Optimize both camera positions and 3D point locations:
```
# Simulated data with noise
3 camera matrices (3x4 each)
4 3D points
12 2D observations (noisy)
```

### Implementation Tasks
```python
# 1. Linearize the non-linear projection equations
# 2. Set up large sparse system for all parameters
# 3. Use iterative solver (least squares with damping)
# 4. Handle the Jacobian matrix structure
# 5. Compare before/after reprojection errors
# 6. Visualize 3D scene and cameras
```

### Learning Outcomes
- Understand large-scale optimization
- Learn sparse linear systems
- Handle iterative refinement

---

## Assignment 9: Robust Plane Fitting (RANSAC)
**Difficulty**: ⭐⭐⭐⭐  
**Time**: 2 hours  
**Goal**: Handle outliers in linear systems

### Problem
Fit a 3D plane to point cloud data containing 70% inliers and 30% outliers:
```
# Generate synthetic data: plane + noise + outliers
100 3D points total
Plane equation: ax + by + cz + d = 0
```

### Implementation Tasks
```python
# 1. Implement basic least squares plane fitting
# 2. Add RANSAC framework for robust estimation
# 3. In each iteration, solve 3-point plane system
# 4. Count inliers using distance threshold
# 5. Refine final solution with all inliers
# 6. Compare robust vs non-robust results
```

### Learning Outcomes
- Understand robust estimation
- Combine RANSAC with linear algebra
- Handle real-world noisy data

---

## Assignment 10: Structure from Motion Initialization
**Difficulty**: ⭐⭐⭐⭐⭐  
**Time**: 3 hours  
**Goal**: Complete computer vision pipeline

### Problem
Initialize 3D structure and camera motion from two uncalibrated images:
```
# Feature correspondences between two images
50+ point correspondences
Unknown camera intrinsics
Unknown relative pose
```

### Implementation Tasks
```python
# 1. Estimate fundamental matrix from correspondences
# 2. Extract essential matrix (assume known intrinsics first)
# 3. Decompose essential matrix to R,t
# 4. Triangulate initial 3D points
# 5. Handle the 4-fold ambiguity in pose
# 6. Bundle adjust the initial reconstruction
# 7. Evaluate reconstruction quality
```

### Learning Outcomes
- Integrate multiple linear systems
- Understand complete SfM pipeline
- Handle geometric ambiguities
- Master the full workflow

---

## Progressive Learning Path

### **Days 3-4 Schedule** (16 hours total):
- **Hour 1-2**: Assignments 1-2 (basic systems)
- **Hour 3-5**: Assignments 3-4 (homogeneous + transformations)  
- **Hour 6-9**: Assignments 5-6 (camera + homography)
- **Hour 10-13**: Assignments 7-8 (3D reconstruction)
- **Hour 14-16**: Assignments 9-10 (robust methods + integration)

### **Skills Progression**:
1. **Basic mechanics** → NumPy operations, verification
2. **System types** → Overdetermined, homogeneous, rank-deficient
3. **Geometric meaning** → Transformations, projections
4. **CV applications** → Cameras, stereo, structure from motion
5. **Real-world issues** → Noise, outliers, numerical stability

### **Key Concepts Covered**:
- Matrix decompositions (LU, SVD, QR)
- Least squares vs exact solutions
- Homogeneous vs inhomogeneous systems
- Geometric transformations
- Multi-view geometry fundamentals
- Robust estimation principles

### **Tools You'll Master**:
```python
# Essential NumPy functions
np.linalg.solve()      # Basic solver
np.linalg.lstsq()      # Least squares
np.linalg.svd()        # Singular value decomposition
np.linalg.cond()       # Condition number
np.linalg.matrix_rank() # Rank computation
np.linalg.norm()       # Various norms
```

### **Success Criteria**:
- Solve any assignment in < 30 minutes on second attempt
- Understand when to use each solver type
- Recognize linear systems in CV problems
- Debug numerical issues confidently
- Visualize results to verify correctness
"""