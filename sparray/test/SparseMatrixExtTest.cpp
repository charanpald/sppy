
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/DynamicSparseMatrixExt.h"

using Eigen::SparseMatrix;


void testScalarMultiply() { 
    DynamicSparseMatrixExt<double, Eigen::ColMajor> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    m.insert(2,0) = -1;    
    m.insert(2,2) = 5.12;
    
    DynamicSparseMatrixExt<double, Eigen::ColMajor> m2(m);
    m.scalarMultiply(2.0);

    m.printValues();
    m2.printValues();
}

void testMatrixAdd() { 
    DynamicSparseMatrixExt<double, Eigen::ColMajor> m(3,3);
    m.insert(0,0) = 3;
    m.insert(1,0) = 2.5;
    m.insert(0,1) = -1;
    m.insert(2,0) = -1;    
    m.insert(2,2) = 5.12;

    DynamicSparseMatrixExt<double, Eigen::ColMajor> m2(3, 3);
    m2.insert(0, 0) = 2;
    
    DynamicSparseMatrixExt<double> m3(3, 3);

    m.printValues();
    m3.printValues();

    m = m2; 
    m.printValues();

    m3 = m + m2; 
    
}


int main()
{


    testMatrixAdd();

}