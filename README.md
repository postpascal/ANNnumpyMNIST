> All bellow equations are only based on one sample for simplication, the index $i,j$ refer to different dimensions.

```math

Softmax=f(z_i)=\frac{e^{z_i}}{\sum\limits_{i=1}^c e^{z_i}}=y_i
```

\begin{equation*}
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
\end{equation*}

\begin{aligned}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\   \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0 
\end{aligned}


### Derivative of Softmax

$$y_j=\frac{e^{z_j}}{\sum\limits_{i=j}^c e^{z_j}}$$

if i=j
$$
\begin{align*}
\frac{\partial y_j}{\partial z_i}&=\frac{e^{z_i}\sum\limits_{i=1}^c e^{z_i}-e^{z_i}e^{z_i}}{(\sum\limits_{i=1}^c e^{z_i})^2}\\
&=y_j(1-y_j)
\end{align*}$$
if i$\neq$ j
$$
\begin{align*}
\frac{\partial y_j}{\partial z_i}&=\frac{0-e^{z_i}e^{z_j}}{(\sum\limits_{i=1}^c e^{z_i})^2}\\
&=-y_iy_j
\end{align*}$$

So its Jacobian matrix is $diag(\hat y)-\hat y\hat y^T$



### Derivative of Cross Entropy

$$CE= -\sum_{i=1}^MY_{i}\log(y_{i})$$

$$\frac{\partial CE}{\partial y_i}=-\frac{1}{y_i}, if Y_i=1$$

### Derivative of Weights
$$\begin{array}{l l l} \frac{\partial l}{\partial w} & = & \frac{\partial l}{\partial y}\frac{\partial y}{\partial z}\frac{\partial z}{\partial w} \\ & = & -\frac{1}{y_j}\langle -y_1y_j,-y_2y_j,\cdots,y_j(1-y_j),-y_iy_j\rangle x \\ & = & (y-Y)x \\  \end{array}$$

### Gradient Desent for Weights Optimization

> Taylor's Equation in $w_0$ for first stage:

$$L(w)=L(w_0)+L^{'}(w_0)(w-w_0)$$
Can rewrite as :
$$L(w_t)=L(w_{t-1})+L^{'}(w_{t-1})(w_t-w_{t-1})$$

$$L(w_t)=L(w_{t-1})+L^{'}(w_{t-1})\Delta w$$

To make $L(w_t)<L(w_t-1{})$
$$\Delta w=-L^{'}(w_{t-1})$$

So
$$w_t=w_{t-1}+\Delta w=w_{t-1}-L^{'}(w_{t-1})$$