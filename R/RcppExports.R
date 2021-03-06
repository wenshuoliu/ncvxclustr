# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#'From the results of the RANN package, calculate the edge indices, the corresponding distances,
#'   and the edge incidence matrix Phi
#'
#'@param idx,dist the output of RANN::nn2()
#'
#'@return a list consist of: a 2*nEdges (# of edges) matrix of indices of edges; a nEdges-vector
#'   dists of the corresponding distances; a nEdges*n edge incidence matrix Phi
#'
#'@export
weights_init <- function(idx, dist) {
    .Call('ncvxclustr_weights_init', PACKAGE = 'ncvxclustr', idx, dist)
}

obj_dual <- function(XF2, U) {
    .Call('ncvxclustr_obj_dual', PACKAGE = 'ncvxclustr', XF2, U)
}

obj_primal <- function(Delta, V, weights) {
    .Call('ncvxclustr_obj_primal', PACKAGE = 'ncvxclustr', Delta, V, weights)
}

proj_l2_acc <- function(Lambda, radii) {
    .Call('ncvxclustr_proj_l2_acc', PACKAGE = 'ncvxclustr', Lambda, radii)
}

#'Solve the projected dual ascent problem with fixed weights
#'
#'@param X the data, with the columns being units, the rows being features
#'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
#'@param weights the non-zero weights in a vector
#'@param Lambda0 the initial guess of Lambda
#'@param maxiter maximum iterations
#'@param eps the duality gap tolerence
#'@param nv step size
#'@param trace whether print out the iterations
#'
#'@return a list including U, V, Lambda and number of iterations to convergence
#'
#'@export
dual_ascent <- function(X, Phi, weights, Lambda0, maxiter, eps, nv, trace) {
    .Call('ncvxclustr_dual_ascent', PACKAGE = 'ncvxclustr', X, Phi, weights, Lambda0, maxiter, eps, nv, trace)
}

#'Solve the projected dual ascent problem with fixed weights and adaptive step size
#'
#'@param X the data, with the columns being units, the rows being features
#'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
#'@param weights the non-zero weights in a vector
#'@param Lambda0 the initial guess of Lambda
#'@param maxiter maximum iterations
#'@param eps the duality gap tolerence
#'@param nv initial step size
#'@param trace whether save the primal and dual values of every iteration
#'
#'@return a list including U, V, Lambda and number of iterations to convergence
#'
#'@export
dual_ascent_adapt <- function(X, Phi, weights, Lambda0, maxiter, eps, nv0, trace) {
    .Call('ncvxclustr_dual_ascent_adapt', PACKAGE = 'ncvxclustr', X, Phi, weights, Lambda0, maxiter, eps, nv0, trace)
}

#'Solve the projected dual ascent problem with fixed weights and adaptive step size and back-tracking
#'
#'@param X the data, with the columns being units, the rows being features
#'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
#'@param weights the non-zero weights in a vector
#'@param Lambda0 the initial guess of Lambda
#'@param maxiter maximum iterations
#'@param eps the duality gap tolerence
#'@param nv initial step size
#'@param trace whether save the primal and dual values of every iteration
#'
#'@return a list including U, V, Lambda and number of iterations to convergence
#'
#'@export
dual_ascent_fasta <- function(X, Phi, weights, Lambda0, maxiter, eps, nv0, trace) {
    .Call('ncvxclustr_dual_ascent_fasta', PACKAGE = 'ncvxclustr', X, Phi, weights, Lambda0, maxiter, eps, nv0, trace)
}

#'@export
mcp_prime <- function(v_norms, lambda, gamma) {
    .Call('ncvxclustr_mcp_prime', PACKAGE = 'ncvxclustr', v_norms, lambda, gamma)
}

#'@export
mcp <- function(v_norms, lambda, gamma) {
    .Call('ncvxclustr_mcp', PACKAGE = 'ncvxclustr', v_norms, lambda, gamma)
}

#'Solve the fusion clustering problem with MCP penalization via MM algorithm
#'
#'@param X the data, with the columns being units, the rows being features
#'@param U0 the initial guess of the centroid matrix U
#'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
#'@param lambda,gamma the parameters of MCP penalty function
#'@param maxiter maximum iterations
#'@param tol the duality gap tolerence
#'@param trace whether save the primal and dual values of every iteration
#'
#'@return a list containing the solution U, V, and (optional) trace information
#'@export
fusion_cluster <- function(X, U0, Phi, lambda, gamma, maxiter_mm, maxiter_cvx, tol, trace) {
    .Call('ncvxclustr_fusion_cluster', PACKAGE = 'ncvxclustr', X, U0, Phi, lambda, gamma, maxiter_mm, maxiter_cvx, tol, trace)
}

