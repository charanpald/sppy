
#ifndef SparseMatrixEXT_H
#define SparseMatrixEXT_H
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include <vector> 

using Eigen::SparseMatrix;

template <class T, int S=Eigen::ColMajor>
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
    void nonZeroInds(long* array1, long* array2) { 
        int i = 0; 

        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            array1[i] = it.row(); 
            array2[i] = it.col();
            i++; 
            }  
        }
    }

    void nonZeroVals(T* array) { 
        int i = 0; 

        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            array[i] = it.value(); 
            i++; 
            }  
        }
    }

    std::vector<long> getIndsRow(int row) { 
        std::vector<long> inds;

        for (int k=0; k<this->outerSize(); ++k) {
          for (typename SparseMatrixExt<T>::InnerIterator it(*this,k); it; ++it) {
            if (it.row() == row) { 
                inds.insert(inds.end(), it.col()); 
                }
            }  
        }

        return inds; 
    }

    std::vector<long> getIndsCol(int col) { 
        std::vector<long> inds;

      for (typename SparseMatrixExt<T>::InnerIterator it(*this, col); it; ++it) {
            inds.insert(inds.end(), it.row());
        }

        return inds; 
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

    SparseMatrixExt<T, S> dot(const SparseMatrixExt& other) { 
        return (SparseMatrixExt<T, S>)((*this) * other); 
        }

    SparseMatrixExt<T, S> add(const SparseMatrixExt& other) { 
        return (SparseMatrixExt<T, S>)((*this) + other); 
        }
    
    SparseMatrixExt<T, S> subtract(SparseMatrixExt const& other) { 
        return ((SparseMatrixExt<T, S>)((*this) - other)); 
        }

    SparseMatrixExt<T, S> trans() { 
        SparseMatrix<T, S> A = this->transpose();
        return (SparseMatrixExt<T, S>)A; 
        }

    SparseMatrixExt<T, S> negate() { 
        SparseMatrix<T, S> A = -(*this);
        return (SparseMatrixExt<T, S>)A; 
        }

    SparseMatrixExt<T, S> abs() { 
        SparseMatrix<T, S> A = this-> cwiseAbs();
        return (SparseMatrixExt<T, S>)A; 
        }
    
    SparseMatrixExt<T, S> hadamard(SparseMatrixExt const& other) { 
        return (SparseMatrixExt<T, S>)this->cwiseProduct(other); 
        }

    void fill(T val) { 
        this->reserve(this->rows()*this->cols());
        for (int i=0; i<this->rows(); i++) 
            for (int j=0; j<this->cols(); j++) 
                this->coeffRef(i, j) = val;
        }

  };

#endif

