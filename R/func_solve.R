
#'Function to calculate the clustering path
#'
#'@param X the data, with the columns being units, the rows being features
#'@param lambda a vector of the tuning parameter
#'@param gamma a scaler of the MCP tuning parameter
#'@param nnn number of nearest neighbors for the preprocessing
#'@param tol.nn nearest neighbor tolerance for the preprocessing
#'
#'@importFrom RANN nn2
#'
#'@return the list of the centroids and the list their differences, for each lambda
#'
#'@export
fusion_cluster_path <- function(X, lambda, gamma, nnn, tol.nn = 0., maxiter.mm = 200,
                         maxiter.cvx = 30000, tol = 1e-5)
{
  nn <- nn2(t(X), k = nnn + 1, eps = 0)
  init <- weights_init(nn$nn.idx, nn$nn.dists)
  Phi <- init$Phi
  nEdge <- dim(Phi)[1]
  nlambda <- length(lambda)
  min.dist <- min(init$dists)
  if(min(lambda) < min.dist){
    cat("The smallest lambda should be larger than", min.dist/gamma,
        "\n When lambda is smaller than that, the MM iteration cannot start.\n")
  }
  U.list <- vector("list", length = nlambda)
  V.list <- vector("list", length = nlambda)
  U = X
  for(j in 1:nlambda){
    mm <- fusion_cluster(X, U, Phi, lambda[j], gamma, maxiter.mm, maxiter.cvx, tol, FALSE)
    U <- mm$U
    U.list[[j]] <- U
    V.list[[j]] <- mm$V
    v_norms <- apply(mm$V, 2, function(x){sqrt(sum(x^2))})
    if(max(v_norms) < 0.1*tol){
      cat("All centriods collapse when lambda >=", lambda[j], "\n")
      U.list <- U.list[1:j]
      V.list <- V.list[1:j]
      break
    }
  }
  path <- list(U = U.list, V = V.list)
  class(path) <- "path"
  return(path)
}
