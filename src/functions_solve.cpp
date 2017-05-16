#include <RcppEigen.h>
using namespace Rcpp;

//[[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;
using Eigen::Lower;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Trip;
typedef Eigen::Map<MatrixXd> MapMatd;

class Edge
{
public:
  explicit Edge(const int i, const int j, const double vij){
    if(i>j){
      l1 = j;
      l2 = i;
    }else{
      l1 = i;
      l2 = j;
    }
    val = vij;
  }

  int head() const {
    return l1;
  }

  int tail() const {
    return l2;
  }

  double value() const{
    return val;
  }

  bool operator<(const Edge & rhs) const
  {
    if(l1!= rhs.l1){
      return l1<rhs.l1;
    }else{
      return l2<rhs.l2;
    }
  }
private:
  int l1, l2;
  double val;
};

//'From the results of the RANN package, calculate the edge indices, the corresponding distances,
//'   and the edge incidence matrix Phi
//'
//'@param idx,dist the output of RANN::nn2()
//'
//'@return a list consist of: a 2*nEdges (# of edges) matrix of indices of edges; a nEdges-vector
//'   dists of the corresponding distances; a nEdges*n edge incidence matrix Phi
//'
//'@export
//[[Rcpp::export]]
List weights_init(const MatrixXi & idx, const MatrixXd dist)
{
  int n = idx.rows();
  int k = idx.cols() - 1; //number of nearest neighbors

  std::set<Edge> edgeSet;
  for(int i=0; i < n; i++){
    for(int j = 1; j < k+1; j++){
      edgeSet.insert(Edge(idx(i, 0), idx(i, j), dist(i, j)));
    }
  }
  int nEdge = edgeSet.size();

  SpMat Phi(nEdge, n);
  MatrixXi edgeIdx(2, nEdge);
  VectorXd dists(nEdge);

  int l=0;
  std::set<Edge>::iterator it;
  for(it = edgeSet.begin(); it != edgeSet.end(); ++it){
    edgeIdx.col(l) << it->head(), it->tail();
    dists(l) = it->value();
    l++;
  }

  std::vector<Trip> Tlist(2*nEdge);
  for(int i=0; i<nEdge; i++){
    Tlist.push_back(Trip(i, edgeIdx(0, i)-1, 1.)); //edgeIdx starts from 1, but index of Phi starts from 0
    Tlist.push_back(Trip(i, edgeIdx(1, i)-1, -1.));
  }
  Phi.setFromTriplets(Tlist.begin(), Tlist.end());

  return List::create(Named("edges") = edgeIdx, Named("dists") = dists,
                      Named("Phi") = Phi);
}

//Objective function for Lagrangian dual of convex clustering problem.
//dual(Lambda) = -0.5||X+Lambda*Phi||_F^2 + 0.5||X||_F^2
//= -0.5||U||_F^2 + 0.5||X||_F^2
//[[Rcpp::export]]
double obj_dual(const double XF2, const MatrixXd & U){
  return(0.5*XF2 - 0.5*U.squaredNorm());
}

//Objective function of convex clustering problem.
//primal(U) = 0.5||X - U||_F^2 + J(U) = 0.5*||Delta||_F^2 + J(V)
//[[Rcpp::export]]
double obj_primal(const MatrixXd & Delta, const MatrixXd & V, const VectorXd & weights){
  int nEdge = V.cols();
  double obj = 0.5*Delta.squaredNorm();

  double penal = 0;
  for(int i=0; i<nEdge; i++){
    penal += V.col(i).norm()*weights(i); //weights includes gamma
  }
  return(obj+penal);
}

//Euclidean projection of the columns of Lambda to the balls with different radii
//  This function is a mutator in C++, but when export to R it doesn't do anything.
void proj_l2_mut(MatrixXd & Lambda, const VectorXd & radii){
  int nEdge = Lambda.cols();
  for(int i=0; i<nEdge; i++){
    double norm = Lambda.col(i).norm();
    if(norm > radii(i)){
      Lambda.col(i) *= (radii(i)/norm);
    }
  }
}

//The accessor version of the proj_l2_mut() function.
//[[Rcpp::export]]
MatrixXd proj_l2_acc(const MatrixXd & Lambda, const VectorXd & radii){
  MatrixXd out = Lambda;
  //Rcout<<out<<std::endl;
  proj_l2_mut(out, radii);
  //Rcout<<out<<std::endl;
  return(out);
}

//test for mutator
//Conclusion: only the standard types from R (NumericMatrix, NumericVector, etc.) are mutable. To use
//  the eigen functionalities, one needs to use the Map<> template.
///[[Rcpp::export]]
void mutate(NumericMatrix & M){
  int ncol = M.cols();
  int nrow = M.rows();
  double *p = &M(0, 0);
  MapMatd Mmap(p, nrow, ncol);
  Mmap.col(0) = VectorXd::Zero(nrow);
}

//'Sovle the projected dual ascent problem with fixed weights
//'
//'@param X the data, with the columns being units, the rows being features
//'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
//'@param weights the non-zero weights in a vector
//'@param Lambda0 the initial guess of Lambda
//'@param maxiter maximum iterations
//'@param eps the duality gap tolerence
//'@param nv step size
//'@param trace whether print out the iterations
//'
//'@return a list including U, V, Lambda and number of iterations to convergence
//'
//'@export
//[[Rcpp::export]]
List dual_ascent(const MatrixXd & X, const SpMat & Phi, const VectorXd & weights,
                     const MatrixXd Lambda0, int maxiter, double eps, double nv,
                     bool trace){
  double XF2 = X.squaredNorm(); //avoid repeatedly doing this calculation
  MatrixXd Delta = Lambda0*Phi;
  MatrixXd U = X + Delta;
  MatrixXd V = U*Phi.transpose();
  double dual = obj_dual(XF2, U);
  double primal = obj_primal(Delta, V, weights);
  MatrixXd Lambda = Lambda0;
  MatrixXd grad(Lambda0.cols(), Lambda0.rows()); //initialize gradient of Lambda with its dimension
  int it = 0;
  VectorXd primal_trace(maxiter);
  VectorXd dual_trace(maxiter);

  while((primal - dual) > eps && it < maxiter){
    grad = -V;
    Lambda += nv*grad; //gradient ascent
    proj_l2_mut(Lambda, weights); //projection on the constraining balls

    Delta = Lambda*Phi;
    U = X + Delta;
    V = U*Phi.transpose();
    primal = obj_primal(Delta, V, weights);
    dual = obj_dual(XF2, U);
    if(trace){
      primal_trace(it) = primal;
      dual_trace(it) = dual;
    }
    it += 1;
  }
  if(it == maxiter) {
    Rcout<<"Projected dual ascent doesn't converge! Try increase maxiter."<<std::endl;
    Rcout<<"Duality gap residual: "<<(primal - dual)<<std::endl;
  }else{
    Rcout<<"Projected dual ascent converged at iteration "<<it<<std::endl;
    if(trace){
      primal_trace.conservativeResize(it);
      dual_trace.conservativeResize(it);
    }
  }
  if(trace) return List::create(Named("U") = U, Named("V") = V, Named("Lambda") = Lambda,
     Named("primal_trace") = primal_trace, Named("dual_trace") = dual_trace);
  else return List::create(Named("U") = U, Named("V") = V, Named("Lambda") = Lambda);
}

//test for matrix inner product
//Conclusion: there's no cognizable difference between treating the matrices as vectors, and trace(M2^T*M1)
///[[Rcpp::export]]
double matrix_inner(const MatrixXd & M1, const MatrixXd & M2, bool vec){
  if(vec){
    const double * p1 = & M1(0, 0);
    const double * p2 = & M2(0, 0);
    Eigen::Map<const VectorXd> v1(p1, M1.size());
    Eigen::Map<const VectorXd> v2(p2, M2.size());
    return v1.dot(v2);
  }else{
    return (M2.transpose()*M1).trace();
  }
}

//'Solve the projected dual ascent problem with fixed weights and adaptive step size
//'
//'@param X the data, with the columns being units, the rows being features
//'@param Phi the edge incidence matrix, defined as Phi_{li} = 1 if(l_1 == i); -1 if(l_2 == i); 0 otherwise
//'@param weights the non-zero weights in a vector
//'@param Lambda0 the initial guess of Lambda
//'@param maxiter maximum iterations
//'@param eps the duality gap tolerence
//'@param nv initial step size
//'@param trace whether save the primal and dual values of every iteration
//'
//'@return a list including U, V, Lambda and number of iterations to convergence
//'
//'@export
//[[Rcpp::export]]
List dual_ascent_adapt(const MatrixXd & X, const SpMat & Phi, const VectorXd & weights,
                 const MatrixXd Lambda0, int maxiter, double eps, double nv0,
                 bool trace){
  double XF2 = X.squaredNorm(); //avoid repeatedly doing this calculation
  MatrixXd Delta = Lambda0*Phi;
  MatrixXd U = X + Delta;
  MatrixXd V = U*Phi.transpose();
  double dual = obj_dual(XF2, U);
  double primal = obj_primal(Delta, V, weights);
  MatrixXd Lambda(Lambda0.cols(), Lambda0.rows());
  //MatrixXd grad(Lambda0.cols(), Lambda0.rows()); //initialize gradient of Lambda with its dimension
  MatrixXd old_Lambda = Lambda0;
  MatrixXd old_grad = -V; //initialize two variable to track the last iteration
  int it = 0;
  VectorXd primal_trace(maxiter);
  VectorXd dual_trace(maxiter);
  double nv = nv0;

  while((primal - dual) > eps && it < maxiter){
    Lambda = old_Lambda + nv*old_grad; //gradient ascent
    proj_l2_mut(Lambda, weights); //projection on the constraining balls

    Delta = Lambda*Phi;
    U = X + Delta;
    V = U*Phi.transpose(); //since grad_Lambda = -V, no need for variable grad
    primal = obj_primal(Delta, V, weights);
    dual = obj_dual(XF2, U);
    if(trace){
      primal_trace(it) = primal;
      dual_trace(it) = dual;
    }

    //calculate the step size of the next iteration
    VectorXd dLambda = Lambda - old_Lambda;
    VectorXd dgrad = -V - old_grad; //grad = -V
    double d_dot_dgrad = (dgrad.transpose()*dLambda).trace();//inner product of dLambda and dgrad
    double nv_s = - dLambda.squaredNorm()/d_dot_dgrad; //for gradient ascent, need a minus sign here
    double nv_m = - d_dot_dgrad/dgrad.squaredNorm();
    if(nv_s>0 && nv_m>0){ //update stepsize nv following Barzilai-Borwein, only when nv_s and nv_m are both positive
      if(nv_m > 0.5*nv_s) nv = nv_m;
      else nv = nv_s - 0.5*nv_m;
    }

    old_Lambda = Lambda;
    old_grad = -V;
    it += 1;
  }
  if(it == maxiter) {
    Rcout<<"Projected dual ascent doesn't converge! Try increase maxiter."<<std::endl;
    Rcout<<"Duality gap residual: "<<(primal - dual)<<std::endl;
  }else{
    Rcout<<"Projected dual ascent converged at iteration "<<it<<std::endl;
    if(trace){
      primal_trace.conservativeResize(it);
      dual_trace.conservativeResize(it);
    }
  }
  if(trace) return List::create(Named("U") = U, Named("V") = V, Named("Lambda") = Lambda,
     Named("primal_trace") = primal_trace, Named("dual_trace") = dual_trace);
  else return List::create(Named("U") = U, Named("V") = V, Named("Lambda") = Lambda);
}
