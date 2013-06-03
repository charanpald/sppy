
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"
#include <vector> 
#include <stdlib.h>  

using Eigen::SparseMatrix;
using Eigen::Triplet;

void testNonZeroInds() { 
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
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsRow(1);
    
    for(unsigned int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testGetIndsCol() { 
    SparseMatrixExt<double, Eigen::ColMajor> A(7, 7); 
    A.reserve(10);

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::vector<long> inds = A.getIndsCol(3);
    
    for(unsigned int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testPutSorted() { 
    int m = 1000000; 
    int n = 1000000;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    const static int numVals = 100000000;  
    
    int i;  
    long *rowInds = new long[numVals]; 
    long *colInds= new long[numVals]; 
    double *vals= new double[numVals];

    
    for(i=0;i<numVals;i++) {
        rowInds[i] =  (long)(rand() % m);
        colInds[i] = (long)(rand() % n);
        vals[i] = (double)((double)rand())/RAND_MAX;
        //std::cout << rowInds[i] << " " << colInds[i] << " " << vals[i] << std::endl;
        }
    
    A.putUsingTriplets(rowInds, colInds, vals, numVals);

    //A.printValues();

    std::cout << A.nonZeros() << std::endl;
    }
    
void testDot() { 
    const static int m = 10; 
    const static int n = 10;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    const static int numVals = 10;  
    
    int i, j;  
    long *rowInds = new long[numVals]; 
    long *colInds= new long[numVals]; 
    double *vals= new double[numVals];
    const static int p = 5;
    double *array = new double[n*p];
    double *result = new double[m*p];
    
    for(i=0;i<n;i++) { 
		for(j=0;j<p;j++) { 
			array[i*p + j] = (double)((double)rand())/RAND_MAX;
			std::cout << array[i*p + j] << " "; 
			if (j==p-1) { 
				std::cout << std::endl; 
			} 
		} 
	} 

    
    for(i=0;i<numVals;i++) {
        rowInds[i] =  (long)(rand() % m);
        colInds[i] = (long)(rand() % n);
        vals[i] = (double)((double)rand())/RAND_MAX;
        std::cout << rowInds[i] << " " << colInds[i] << " " << vals[i] << std::endl;
        }
    
    A.putUsingTriplets(rowInds, colInds, vals, numVals);	
    
    A.dot(array, p, result);
    
    for(i=0;i<m;i++) { 
		for(j=0;j<p;j++) { 
			std::cout << result[i*p + j] << " "; 
			if (j==p-1) { 
				std::cout << std::endl; 
			} 
		} 
	} 
} 

int main()
{

    testDot();

}
