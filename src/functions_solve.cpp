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

//'Calculate the Gaussian Kernel weights
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
  VectorXd weights(nEdge);

  int l=0;
  std::set<Edge>::iterator it;
  for(it = edgeSet.begin(); it != edgeSet.end(); ++it){
    edgeIdx.col(l) << it->head(), it->tail();
    weights(l) = it->value();
    l++;
  }

  std::vector<Trip> Tlist(2*nEdge);
  for(int i=0; i<nEdge; i++){
    Tlist.push_back(Trip(i, edgeIdx(0, i)-1, 1.)); //edgeIdx starts from 1, but index of Phi starts from 0
    Tlist.push_back(Trip(i, edgeIdx(1, i)-1, -1.));
  }
  Phi.setFromTriplets(Tlist.begin(), Tlist.end());

  return List::create(Named("edges") = edgeIdx, Named("weights") = weights,
                      Named("Phi") = Phi);
}
