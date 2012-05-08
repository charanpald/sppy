
#ifndef DYNAMICSPARSEMATRIXEXT_H
#define DYNAMICSPARSEMATRIXEXT_H
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <iostream>
#include <Eigen/Sparse>

using Eigen::DynamicSparseMatrix;

template <class T>
class DynamicSparseMatrixExt:public DynamicSparseMatrix<T> {
  public:
	DynamicSparseMatrixExt<T>(): 
		DynamicSparseMatrix<T>(){ 
		} 

	DynamicSparseMatrixExt<T>(int rows, int cols): 
		DynamicSparseMatrix<T>(rows, cols){ 
		}
    
    void insertVal(int row, int col, T val) { 
        this->coeffRef(row, col) = val;
        }
  };

#endif

