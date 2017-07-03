library(ncvxclustr)
library(RANN)
library(microbenchmark)
library(Matrix)


load("~/Codes/Rproj/cvxclustr/data/mammals.rdata")

p=2;n=200;nnn=5;

set.seed(123)
Xt <- matrix(rnorm(p*n),n,p)
X <- t(Xt)
nn <- nn2(Xt, k = nnn + 1, eps = 0)

init <- weights_init(nn$nn.idx, nn$nn.dists)
Phi <- init$Phi
nEdge <- dim(Phi)[1]

gk_weights <- exp(-1.0*(init$dists)^2)
gamm <- 50.

Lambda0 <- matrix(rnorm(p*nEdge), p, nEdge)
#Lambda <- proj_l2_acc(Lambda0, radii = gk_weights)

####calculate step size from the package cvxclustr####
weights.cvx <- cvxclustr::kernel_weights(X, phi = 1.0)
weights.cvx <- cvxclustr::knn_weights(weights.cvx, nnn, n)
step.size <- cvxclustr::AMA_step_size(weights.cvx, n)

res.ncvx <- dual_ascent(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 10000, eps = 1e-2, nv = step.size, trace = TRUE)
res.ncvx <- dual_ascent_adapt(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 10000, eps = 1e-3, nv0 = step.size, trace = TRUE)

microbenchmark(res.ncvx <- dual_ascent_adapt(X, Phi, weights = gamm*gk_weights, Lambda0, maxiter = 30000, eps = 1e-3, nv = 00.1,
                                     trace = TRUE), times = 1)

# testing of the ncvx clustering with MCP penalty ----
gamma <- 1.0
maxiter.cvx <- 30000
maxiter.mm <- 200
mm <- fusion_cluster(X, X, Phi, 5.8, gamma, maxiter_mm = maxiter.mm, maxiter_cvx = maxiter.cvx,
                     tol = 1e-4, trace = TRUE)
v.end <- apply(mm$V, 2, function(x){sqrt(sum(x^2))})
max(v.end)

lambda <- seq(min(init$dists), 5.8, length.out = 40)
microbenchmark(path <- fusion_cluster_path(X, lambda, gamma, nnn, tol.nn = 0., maxiter.mm, maxiter.cvx,
                            tol = 1e-3), times = 1)
plot_path(X, path)
ggsave("~/Dropbox/Research/SAMSI/fusion_clustering/ncvx_path.jpeg", width = 7, height = 5)

####output of package cvxclustr####
res.cvx <- cvxclustr::cvxclust(X, weights.cvx, gamm, method = "ama", nu = 1/n, tol = 1e-4, max_iter = 10000,
                          type = 2, accelerate = FALSE)
microbenchmark(res.cvx <- cvxclustr::cvxclust(X, weights.cvx, gamm, method = "ama", nu = step.size, tol = 1e-3,
                                              max_iter = 500000, type = 2, accelerate = TRUE), times = 1)
lambda <-seq(0, 7.6, length.out = 40)
microbenchmark(path.cvx <- cvxclustr::cvxclust(X, weights.cvx, lambda, method = "ama", nu = step.size, tol = 1e-3,
                                max_iter = 10000, type = 2, accelerate = TRUE), times = 1)
plot_path(X, path.cvx)
ggsave("~/Dropbox/Research/SAMSI/fusion_clustering/cvx_path.jpeg", width = 7, height = 5)
max(apply(path.cvx$V[[length(lambda)]], 2, function(x){sqrt(sum(x^2))}))
