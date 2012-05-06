
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;

int main()
{
    SparseMatrix<double, Eigen::RowMajor> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    
    for (int k=0; k<m.outerSize(); ++k) {
      for (SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(m,k); it; ++it)
      {
        it.value();
        it.row();   // row index
        it.col();   // col index (here it is equal to k)
        it.index(); // inner index, here it is equal to it.row()

        std::cout << it.row() << ", " <<  it.col() << ", " <<  it.value() << std::endl; 
        
      }
 }

    std::cout << "Printing full matrix" << std::endl; 
    for (int i=0;i<3;i++) { 
        for (int j=0;j<3;j++) { 
            std::cout << i << ", " <<  j << ", " <<  m.coeff(i, j) << std::endl;             
            }        
        }


  //std::cout << m << std::endl;
}