
#ifndef DYNAMICSPARSEMATRIXEXT_H
#define DYNAMICSPARSEMATRIXEXT_H
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <iostream>
#include <Eigen/Sparse>

using Eigen::DynamicSparseMatrix;

template <class T>

//Class is column major 
class DynamicSparseMatrixExt:public DynamicSparseMatrix<T> {
  public:
	DynamicSparseMatrixExt<T>(): 
		DynamicSparseMatrix<T>(){ 
		} 

	DynamicSparseMatrixExt<T>(int rows, int cols): 
		DynamicSparseMatrix<T>(rows, cols){ 
		}
    
    void insertVal(int row, int col, T val) { 
        if (val != 0)
            this->coeffRef(row, col) = val;
        }

    void printValues() { 
        for (int k=0; k<this->outerSize(); ++k) {
          for (DynamicSparseMatrixExt<double>::InnerIterator it(*this,k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") " << it.value() << std::endl;  
            }  
        }
    } 

    //Have function to give nonzero elements by passing in points to arrays 
    //Input array points must have the same size as the number of nonzeros in this matrix
    void nonZeroInds(int* array1, int* array2) { 
        int i = 0; 

        for (int k=0; k<this->outerSize(); ++k) {
          for (DynamicSparseMatrixExt<double>::InnerIterator it(*this,k); it; ++it) {
            array1[i] = it.row(); 
            array2[i] = it.col();
            i++; 
            }  
        }
    }

    DynamicSparseMatrixExt<T>* slice(int* array1, int size1, int* array2, int size2) { 
        //Array indices must be sorted 
        DynamicSparseMatrixExt *mat = new DynamicSparseMatrixExt<T>(size1, size2);
        int size1Ind = 0; 

        //Assume column major class - j is col index 
        for (int j=0; j<size2; ++j) { 
            size1Ind = 0; 
            for (typename DynamicSparseMatrixExt<T>::InnerIterator it(*this, array2[j]); it; ++it) {
                while (array1[size1Ind] < it.row() && size1Ind < size1) { 
                    size1Ind++; 
                    }

                if(it.row() == array1[size1Ind]) { 
                    mat->insert(size1Ind, j) = it.value();                    
                    }

                }    
            }

        return mat;
        }
  };

#endif

