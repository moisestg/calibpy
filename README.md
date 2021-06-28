# calibpy

 [Chrome extension to visualize math in Github Markdown](https://chrome.google.com/webstore/detail/github-math-display/cgolaobglebjonjiblcjagnpmdmlgmda/)

## General description
Tiny camera calibration library written in Python3

## Documentation
`COMING SOON!`

## Homography estimation (Direct Linear Transform algorithm)

Given a cloud of 3D points and their corresponding image points captured by a camera looking at the scene, if we have $N$ points with $N >= 6$, we can resort to the field of projective geometry and use the perspective-and-point algorithm to estimate the projective transformation $P$ that maps 3D points $\mathbf{\overrightarrow{X_i}}$ to 2D image points $\mathbf{\overrightarrow{x_i}}$ (up to a scale factor) [1][1]. In a general setting, they are related by the following linear equation (in homogeneous coordinates [2][2]):

$\lambda \begin{bmatrix}\mathbf{\overrightarrow{x_i}} \\ 1 \end{bmatrix} = K \begin{bmatrix} R_{3\times3} & \mathbf{\overrightarrow{t}}_{3\times1} \\ \mathbf{\overrightarrow{0}}^{T}_{1\times3} & 1 \end{bmatrix}  \begin{bmatrix}\mathbf{\overrightarrow{X_i}} \\ 1 \end{bmatrix} = P \begin{bmatrix}\mathbf{\overrightarrow{X_i}} \\ 1 \end{bmatrix}$

$P=K \begin{bmatrix} R_{3\times3} & \mathbf{\overrightarrow{t}}_{3\times1} \\ \mathbf{\overrightarrow{0}}^{T}_{1\times3} & 1 \end{bmatrix}$

As the cross-product of a vector with itself is equal to 0, we can cross multiply both sides of the above equation to obtain ($\begin{bmatrix}\;\end{bmatrix}_{\times}$ being the matrix formulation of the cross-product with a vector):

$\begin{bmatrix}\mathbf{\overrightarrow{x_i}} \\ 1 \end{bmatrix}_{\times} P \begin{bmatrix}\mathbf{\overrightarrow{X_i}} \\ 1 \end{bmatrix} = \mathbf{\overrightarrow{0}}$

In the case of a planar 3D object, we can choose a coordinate system such that for all points we have $z=0$ (reducing the homogeneous 3D point to the 2D homogeneous point $\mathbf{\overrightarrow{X_i}}^{'}$) and simplify the problem to estimating the $3\times3$ homography matrix $H$:

$\begin{bmatrix}\mathbf{\overrightarrow{x_i}} \\ 1 \end{bmatrix}_{\times} H \; \mathbf{\overrightarrow{X_i}}^{'} = \begin{bmatrix}u_i \\ v_i \\ 1 \end{bmatrix}_{\times} \begin{bmatrix}\mathbf{\overrightarrow{h_1}}^{T} \\ \mathbf{\overrightarrow{h_2}}^{T} \\ \mathbf{\overrightarrow{h_3}}^{T} \end{bmatrix} \mathbf{\overrightarrow{X_i}}^{'} = \begin{bmatrix}u_i \\ v_i \\ 1 \end{bmatrix}_{\times} \begin{bmatrix}\mathbf{\overrightarrow{h_1}}^{T} \mathbf{\overrightarrow{X_i}}^{'} \\ \mathbf{\overrightarrow{h_2}}^{T} \mathbf{\overrightarrow{X_i}}^{'} \\ \mathbf{\overrightarrow{h_3}}^{T} \mathbf{\overrightarrow{X_i}}^{'} \end{bmatrix} = \begin{bmatrix}u_i \\ v_i \\ 1 \end{bmatrix}_{\times} \begin{bmatrix}\mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{h_1}}\\ \mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{h_2}} \\ \mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{h_3}} \end{bmatrix} = \begin{bmatrix}0 & -1 & v_i \\ 1 & 0 & -u_i \\ -v_i & u_i & 0 \end{bmatrix}_{\times} \begin{bmatrix}\mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{0}}_{1\times3} \mathbf{\overrightarrow{0}}_{1\times3} \\ \mathbf{\overrightarrow{0}}_{1\times3} \mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{0}}_{1\times3} \\ \mathbf{\overrightarrow{0}}_{1\times3} \mathbf{\overrightarrow{0}}_{1\times3} \mathbf{\overrightarrow{X_i}}^{'T} \end{bmatrix} \begin{bmatrix} \mathbf{\overrightarrow{h_1}} \\ \mathbf{\overrightarrow{h_2}} \\ \mathbf{\overrightarrow{h_3}} \end{bmatrix} = \begin{bmatrix} \mathbf{\overrightarrow{0}}_{1\times3} \mathbf{-\overrightarrow{X_i}}^{'T} v_i \mathbf{\overrightarrow{X_i}}^{'T} \\ \mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{0}}_{1\times3} -u_i\mathbf{\overrightarrow{X_i}}^{'T} \\ -v_i \mathbf{\overrightarrow{X_i}}^{'T} u_i \mathbf{\overrightarrow{X_i}}^{'T} \mathbf{\overrightarrow{0}}_{1\times3} \end{bmatrix} \begin{bmatrix} \mathbf{\overrightarrow{h_1}} \\ \mathbf{\overrightarrow{h_2}} \\ \mathbf{\overrightarrow{h_3}} \end{bmatrix} = \mathbf{\overrightarrow{0}}$

Which has the form of a least-squares problem of the form $A\mathbf{\overrightarrow{h}}=\mathbf{\overrightarrow{0}}$. The unknown $\mathbf{\overrightarrow{h}}$ can be obtained through the singular value decomposition of $A$, $SVD(A)=USV^T$. Then, $\mathbf{\overrightarrow{h}}=V_n$ (the last column of $V$).


## Intrinsic camera matrix estimation

Bla [3][3]




[//]: # (REFERENCES)
[1]: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-120003.3

[2]: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-40003.1

[3]: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-660007.4.1




