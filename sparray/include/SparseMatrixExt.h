
#ifndef SPARSEMATRIXEXT_H
#define SPARSEMATRIXEXT_H
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <iostream>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;

template <class T>
class SparseMatrixRowMaj:public SparseMatrix<T, Eigen::RowMajor> {
  public:
	SparseMatrixRowMaj<T>(): 
		SparseMatrix<T, Eigen::RowMajor>(){ 
		} 

	SparseMatrixRowMaj<T>(int rows, int cols): 
		SparseMatrix<T, Eigen::RowMajor>(rows, cols){ 
		}
    
    void insertVal(int row, int col, T val) { 
        this->insert(row, col) = val;
        }
  };

template <class T>
class SparseMatrixColMaj:public SparseMatrix<T, Eigen::ColMajor> {
  public:
	SparseMatrixColMaj<T>(): 
		SparseMatrix<T, Eigen::ColMajor>(){ 
		} 

	SparseMatrixColMaj<T>(int rows, int cols): 
		SparseMatrix<T, Eigen::ColMajor>(rows, cols){ 
		}
    
    void insertVal(int row, int col, T val) { 
        this->insert(row, col) = val;
        }
  };



#endif

