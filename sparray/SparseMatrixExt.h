
#ifndef DATE_H
#define DATE_H
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <iostream>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;

template <class T>
class SparseMatrixExt:public SparseMatrix<T> {
  public:
	SparseMatrixExt<T>(): 
		SparseMatrix<T>(){ 
		} 

	SparseMatrixExt<T>(int rows, int cols): 
		SparseMatrix<T>(rows, cols){ 
		}
    
    void insertVal(int row, int col, T val) { 
        this->insert(row, col) = val;
        }
  };

#endif

