
#ifndef SparseMatrixEXT_H
#define SparseMatrixEXT_H
#include <iostream>
#include <eigen3/Eigen/Sparse>

using Eigen::SparseMatrix;

template <class T, int S=Eigen::ColMajor>

//Class is column major 
class SparseMatrixExt:public SparseMatrix<T, S> {
  public:
	SparseMatrixExt<T, S>(): 
		SparseMatrix<T, S>(){ 
		} 

	SparseMatrixExt<T, S>(int rows, int cols): 
		SparseMatrix<T, S>(rows, cols){ 
		}


	SparseMatrixExt<T, S>(const SparseMatrix<T, S> other): 
		SparseMatrix<T, S>(other){ 
		}


    SparseMatrixExt& operator=(const SparseMatrixExt& other)  { 
        SparseMatrix<T, S>::operator=(other); 
        return *this;
        }


    /*SparseMatrixExt& operator+(const SparseMatrixExt& other)  { 
        SparseMatrixExt &result; 
        result = this->operator+(other); 
        return *this;
        }*/
    
    void insertVal(int row, int col, T val) { 
        if (this->coeff(row, col) != val)
            this->coeffRef(row, col) = val;
        }

    void printValues() { 
        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            std::cout << "(" << it.row() << ", " << it.col() << ") " << it.value() << std::endl;  
            }  
        }
    } 

    T sumValues() { 
        T result = 0; 
        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            result += it.value();  
            }  
        }

        return result; 
    } 

    //Have function to give nonzero elements by passing in points to arrays 
    //Input array points must have the same size as the number of nonzeros in this matrix
    void nonZeroInds(int* array1, int* array2) { 
        int i = 0; 

        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            array1[i] = it.row(); 
            array2[i] = it.col();
            i++; 
            }  
        }
    }

    void slice(int* array1, int size1, int* array2, int size2, SparseMatrixExt<T, S> *mat) { 
        //Array indices must be sorted 
        //SparseMatrixExt *mat = new SparseMatrixExt<T, S>(size1, size2);
        int size1Ind = 0; 

        //Assume column major class - j is col index 
        for (int j=0; j<size2; ++j) { 
            size1Ind = 0; 
            for (typename SparseMatrixExt<T, S>::InnerIterator it(*this, array2[j]); it; ++it) {
                while (array1[size1Ind] < it.row() && size1Ind < size1) { 
                    size1Ind++; 
                    }

                if(it.row() == array1[size1Ind]) { 
                    mat->insert(size1Ind, j) = it.value();                    
                    }

                }    
            }

        }

    void scalarMultiply(double d) { 
        (*this)*=d; 
        }
    
  };

#endif

