
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <Eigen/Sparse>
#include "include/DynamicSparseMatrixExt.h"

using Eigen::SparseMatrix;

int main()
{
    DynamicSparseMatrixExt<double> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    m.insert(2,0) = -1;    
    m.insert(2,2) = 5.12;

    DynamicSparseMatrixExt<double> m2(3,3);
    m2.insert(0,0) = 3;
    m2.insert(1,0) = 2.5;
    m2.insert(0,1) = -1;
    m2.insert(2,0) = -1;    
    m2.insert(2,2) = 5.12;
    
    //m.printValues();
    //m2.printValues();

    DynamicSparseMatrixExt<double> m3; 
    m3 = m2;
    m2.printValues();
    m3.printValues();

}