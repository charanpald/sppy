
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/DynamicSparseMatrixExt.h"

using Eigen::SparseMatrix;

void testScalarMultiply() { 
    DynamicSparseMatrixExt<double> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    m.insert(2,0) = -1;    
    m.insert(2,2) = 5.12;
    
    DynamicSparseMatrixExt<double> m2(m);
    m.scalarMultiply(2.0);

    m.printValues();
    m2.printValues();

    //DynamicSparseMatrix<double> m2(3,3); 
    //DynamicSparseMatrix<double> m3(m2);
}


int main()
{
    testScalarMultiply();

}