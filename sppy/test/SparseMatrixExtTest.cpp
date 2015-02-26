
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "../include/SparseMatrixExt.h"
#include <vector> 
#include <stdlib.h>  

using namespace Eigen;

void testNonZeroInds() { 
    SparseMatrixExt<double, Eigen::RowMajor> A(7, 7); 
    A.reserve(10);


    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    int rowInds[4]; 
    int colInds[4];

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

    std::vector<int> inds = A.getIndsRow(1);
    
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

    std::vector<int> inds = A.getIndsCol(3);
    
    for(unsigned int i = 0; i<inds.size();i++)
        std::cout << inds[i] << std::endl; 
    
}

void testPutSorted() { 
    int m = 1000000; 
    int n = 1000000;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    const static int numVals = 100000000;  
    
    int i;  
    int *rowInds = new int[numVals]; 
    int *colInds= new int[numVals]; 
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
    int *rowInds = new int[numVals]; 
    int *colInds= new int[numVals]; 
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
    
    A.dot2d(array, p, result);
    
    for(i=0;i<m;i++) { 
		for(j=0;j<p;j++) { 
			std::cout << result[i*p + j] << " "; 
			if (j==p-1) { 
				std::cout << std::endl; 
			} 
		} 
	} 
} 

void testDotSub2d() { 
    const static int m = 10; 
    const static int n = 10;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    const static int numVals = 10;  
    
    int i, j;  
    int *rowInds = new int[numVals]; 
    int *colInds= new int[numVals]; 
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
    
    A.dotSub2d(array, p, 0, 3, result);
    
    for(i=0;i<m;i++) { 
		for(j=0;j<p;j++) { 
			std::cout << result[i*p + j] << " "; 
			if (j==p-1) { 
				std::cout << std::endl; 
			} 
		} 
	} 

}


void testPow() { 
    const static int m = 10; 
    const static int n = 10;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  

    A.coeffRef(0, 1) = 1;
    A.coeffRef(0, 4) = 1.5;
    A.coeffRef(1, 3) = 5.2;
    A.coeffRef(3, 3) = -0.2;

    std::cout << A.norm() << std::endl;
}

void testBiCGSTAB() { 
    const static int m = 4; 
    const static int n = 4;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    VectorXd result(n);
    int info = 3; 

    A.coeffRef(0, 0) = 1;
    A.coeffRef(1, 1) = 2;
    A.coeffRef(2, 2) = 3;
    A.coeffRef(3, 3) = 4;

    double data[] = {1,1,1,1};
    double resultArr[4];

    info = A.biCGSTAB(data, 4, resultArr, 4, 0.0001);

    for(int i=0;i<n;i++) { 
        std::cout << resultArr[i] << std::endl; 
        }
    
    std::cout << std::endl << info << std::endl; 
}

void testPrune() { 
    const static int m = 4; 
    const static int n = 4;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    double eps = 0.0001;

    A.coeffRef(0, 0) = 1;
    A.coeffRef(1, 1) = 2;
    A.coeffRef(2, 2) = -3;
    A.coeffRef(3, 3) = 0;

    std::cout << "Number of nonzeros " << A.nonZeros() << std::endl; 
    A.prune(eps); 
    std::cout << "Number of nonzeros " << A.nonZeros() << std::endl;
    A.printValues();
	
} 

void testBlock() { 
    const static int m = 4; 
    const static int n = 4;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    SparseMatrixExt<double, Eigen::ColMajor> B(2, 2);

    A.coeffRef(0, 0) = 1;
    A.coeffRef(1, 1) = 2;
    A.coeffRef(2, 2) = -3;
    A.coeffRef(3, 3) = 5;

    A.printValues();

    B = A.submatrix(1, 1, 2, 2);

    B.printValues();
}

void testTranpose() { 
    const static int m = 4; 
    const static int n = 4;
    SparseMatrixExt<double, Eigen::ColMajor> A(m, n);  
    SparseMatrixExt<double, Eigen::RowMajor> B(2, 2);

    A.coeffRef(0, 0) = 1;
    A.coeffRef(1, 2) = 2;
    A.coeffRef(2, 3) = -3;
    A.coeffRef(3, 3) = 5;

    //A.printValues();
    B = A.transpose(); 
    
    B.printValues();


}

int main()
{

    testTranpose();

}
