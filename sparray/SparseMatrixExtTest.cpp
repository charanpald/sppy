
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;

int main()
{
  SparseMatrix<double, Eigen::RowMajor> m(2,2);
  m.insert(0,0) = 3;
 m.insert(1,0) = 2.5;
  m.insert(0,1) = -1;

  //m.insert(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}