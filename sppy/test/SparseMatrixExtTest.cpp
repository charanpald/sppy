
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"
#include <vector> 
#include <stdlib.h>  

using Eigen::SparseMatrix;
using Eigen::Triplet;

void testNonZeroInds() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);


    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    long rowInds[4]; 
    long colInds[4];

    A.nonZeroInds(rowInds, colInds);

    for(int i=0;i<4;i++) { 
        std::cout << rowInds[i] << " " << colInds[i] << " " << A.coeffRef(rowInds[i], colInds[i]) << std::endl;         
        }
}

void testSlice() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::ColMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    int rowInds[2] = {0, 1};
    int colInds[3] = {1, 3, 4}; 

    SparseMatrixExt<double, Eigen::ColMajor> B(2, 3);

    A.slice(rowInds, 2, colInds, 3, &B);

    B.printValues();
}

void testGetIndsRow() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsRow(1);
    
    for(int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testGetIndsCol() { 
    int n = 5; 
    SparseMatrixExt<double, Eigen::ColMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsCol(3);
    
    for(int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testPutSorted() { 
    std::cout << "Got here 1" << std::endl;
    int m = 1000000; 
    int n = 1000000;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    const static int numVals = 100000000;  
    
    int i;  
    long *rowInds = new long[numVals]; 
    long *colInds= new long[numVals]; 
    double *vals= new double[numVals];
    
    std::cout << "Got here" << std::endl;
    
    for(i=0;i<numVals;i++) {
        rowInds[i] =  (long)(rand() % m);
        colInds[i] = (long)(rand() % n);
        vals[i] = (double)((double)rand())/RAND_MAX;
        //std::cout << rowInds[i] << " " << colInds[i] << " " << vals[i] << std::endl;
        }
    
    std::cout << "Adding to matrix" << std::endl;
    A.putTriplets(rowInds, colInds, vals, numVals);

    //A.printValues();

    std::cout << A.nonZeros() << std::endl;


    }

int main()
{

    std::cout << "Got here 0" << std::endl;
    testPutSorted();

}
