#'Function to plot the clustering path
#'
#'@param X the data, with the columns being units, the rows being features
#'@param path the output of function \link{fusion_cluster_path}
#'
#'@import ggplot2
#'
#'@export
plot_path <- function(X, path)
{
  svdX <- svd(X)
  pc <- svdX$u[,1:2,drop=FALSE]
  df.paths <- data.frame(x=c(),y=c(), group=c())
  nGamma <- length(path$U)
  for (j in 1:nGamma) {
    pcs <- t(pc) %*% (path$U[[j]])
    x <- pcs[1,]
    y <- pcs[2,]
    df <- data.frame(x=pcs[1,], y=pcs[2,], group=1:n)
    df.paths <- rbind(df.paths,df)
  }
  X_data <- as.data.frame(t(X)%*%pc)
  colnames(X_data) <- c("x","y")
  data_plot <- ggplot(data=df.paths,aes(x=x,y=y))
  data_plot <- data_plot + geom_path(aes(group=group),colour='grey30',alpha=0.5)
  data_plot <- data_plot + geom_point(data=X_data,aes(x=x,y=y),size=1.5)
  data_plot <- data_plot + xlab('Principal Component 1') + ylab('Principal Component 2')
  data_plot + theme_bw()
}
