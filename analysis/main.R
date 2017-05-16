library(ncvxclustr)
library(RANN)
library(microbenchmark)
library(Matrix)


load("~/Codes/Rproj/cvxclustr/data/mammals.rdata")

p=10;n=500;nnn=5;

set.seed(123)
Xt <- matrix(rnorm(p*n),n,p)
X <- t(Xt)
nn <- nn2(Xt, k = nnn + 1, eps = 0)

init <- weights_init(nn$nn.idx, nn$nn.dists)
Phi <- init$Phi
nEdge <- dim(Phi)[1]

gk_weights <- exp(-1.0*(init$dists)^2)
gamm <- 5000.

Lambda0 <- matrix(rnorm(p*nEdge), p, nEdge)
#Lambda <- proj_l2_acc(Lambda0, radii = gk_weights)

####calculate step size from the package cvxclustr####
weights.cvx <- cvxclustr::kernel_weights(X, phi = 1.0)
weights.cvx <- cvxclustr::knn_weights(weights.cvx, nnn, n)
step.size <- cvxclustr::AMA_step_size(weights.cvx, n)

res.ncvx <- dual_ascent(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 10000, eps = 1e-2, nv = step.size, trace = TRUE)
res.ncvx <- dual_ascent_adapt(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 10000, eps = 1e-3, nv0 = step.size, trace = TRUE)

microbenchmark(res.ncvx <- dual_ascent_adapt(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 30000, eps = 1e-3, nv = step.size,
                                     trace = FALSE), times = 1)

####output of package cvxclustr####
res.cvx <- cvxclustr::cvxclust(X, weights.cvx, gamm, method = "ama", nu = 1/n, tol = 1e-4, max_iter = 10000,
                          type = 2, accelerate = FALSE)
microbenchmark(res.cvx <- cvxclustr::cvxclust(X, weights.cvx, gamm, method = "ama", nu = step.size, tol = 1e-3,
                                              max_iter = 500000, type = 2, accelerate = TRUE), times = 1)
