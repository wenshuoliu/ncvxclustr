% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{dual_ascent}
\alias{dual_ascent}
\title{Solve the projected dual ascent problem with fixed weights}
\usage{
dual_ascent(X, Phi, weights, Lambda0, maxiter, eps, nv, trace)
}
\arguments{
\item{X}{the data, with the columns being units, the rows being features}

\item{Phi}{the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise}

\item{weights}{the non-zero weights in a vector}

\item{Lambda0}{the initial guess of Lambda}

\item{maxiter}{maximum iterations}

\item{eps}{the duality gap tolerence}

\item{nv}{step size}

\item{trace}{whether print out the iterations}
}
\value{
a list including U, V, Lambda and number of iterations to convergence
}
\description{
Solve the projected dual ascent problem with fixed weights
}
