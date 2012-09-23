
#ifndef SparseVectorEXT_H
#define SparseVectorEXT_H
#include <iostream>
#include <eigen3/Eigen/Sparse>

using Eigen::SparseVector;

template <class T>
class SparseVectorExt:public SparseVector<T> {
  public:
	SparseVectorExt<T>(): 
		SparseVector<T>(){ 
		} 

	SparseVectorExt<T>(int size): 
		SparseVector<T>(size){ 
		}


	SparseVectorExt<T>(const SparseVector<T> other): 
		SparseVector<T>(other){ 
		}


    SparseVectorExt& operator=(const SparseVectorExt& other)  { 
        SparseVector<T>::operator=(other); 
        return *this;
        }
    
    void insertVal(int ind, T val) { 
        if (this->coeff(ind) != val)
            this->coeffRef(ind) = val;
        }

    void printValues() { 
      for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
        std::cout << "(" << it.index() << ") " << it.value() << std::endl;  
        }  
    } 


    T sumValues() { 
       T result = 0; 
      for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
        result += it.value();  
        }  

        return result; 
    } 


    //Have function to give nonzero elements by passing in points to arrays 
    //Input array points must have the same size as the number of nonzeros in this matrix
    void nonZeroInds(long* array1) { 
        int i = 0; 
        for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
            array1[i] = it.index(); 
            i++; 
        }  
    }

    void slice(int* array1, int size1, SparseVectorExt<T> *mat) { 
        //Array indices must be sorted 
        int size1Ind = 0; 


        for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
            while (array1[size1Ind] < it.index() && size1Ind < size1) { 
                size1Ind++; 
                }

            if(it.index() == array1[size1Ind]) { 
                mat->insert(size1Ind) = it.value();                    
                }
            }    
        }

    SparseVectorExt<T, S> abs() { 
        SparseVector<T, S> A = this-> cwiseAbs();
        return (SparseVectorExt<T, S>)A; 
        }
/*
    void scalarMultiply(double d) { 
        (*this)*=d; 
        }

    SparseVectorExt<T, S> dot(const SparseVectorExt& other) { 
        return (SparseVectorExt<T, S>)((*this) * other); 
        }

    SparseVectorExt<T, S> add(const SparseVectorExt& other) { 
        return (SparseVectorExt<T, S>)((*this) + other); 
        }
    
    SparseVectorExt<T, S> subtract(SparseVectorExt const& other) { 
        return ((SparseVectorExt<T, S>)((*this) - other)); 
        }

    SparseVectorExt<T, S> trans() { 
        SparseVector<T, S> A = this->transpose();
        return (SparseVectorExt<T, S>)A; 
        }

    SparseVectorExt<T, S> negate() { 
        SparseVector<T, S> A = -(*this);
        return (SparseVectorExt<T, S>)A; 
        }


    
    SparseVectorExt<T, S> hadamard(SparseVectorExt const& other) { 
        return (SparseVectorExt<T, S>)this->cwiseProduct(other); 
        }

    void fill(T val) { 
        this->reserve(this->rows()*this->cols());
        for (int i=0; i<this->rows(); i++) 
            for (int j=0; j<this->cols(); j++) 
                this->coeffRef(i, j) = val;
        }*/

  };

#endif

