
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
    
    SparseVectorExt<T> abs() { 
        SparseVector<T> A = this-> cwiseAbs();
        return (SparseVectorExt<T>)A; 
        }

    SparseVectorExt<T> add(const SparseVectorExt& other) { 
        return (SparseVectorExt<T>)((*this) + other); 
        }

    T dot(const SparseVectorExt& other) { 
        return (T)(this->hadamard(other)).sumValues(); 
        }

    void fill(T val) { 
        for (int i=0; i<this->rows(); i++) 
                this->coeffRef(i) = val;
        }

    SparseVectorExt<T> hadamard(SparseVectorExt const& other) { 
        return (SparseVectorExt<T>)this->cwiseProduct(other); 
        }

    void insertVal(int ind, T val) { 
        if (this->coeff(ind) != val)
            this->coeffRef(ind) = val;
        }

    SparseVectorExt<T> negate() { 
        SparseVector<T> A = -(*this);
        return (SparseVectorExt<T>)A; 
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

    void nonZeroVals(T* array) { 
        int i = 0; 

        for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
            array[i] = it.value(); 
            i++; 
            }
        }

    void printValues() { 
      for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
        std::cout << "(" << it.index() << ") " << it.value() << std::endl;  
        }  
    } 

    SparseVectorExt<T> subtract(SparseVectorExt const& other) { 
        return ((SparseVectorExt<T>)((*this) - other)); 
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

    void scalarMultiply(double d) { 
        (*this)*=d; 
        }

    T sumValues() { 
       T result = 0; 
      for (typename SparseVectorExt<T>::InnerIterator it(*this); it; ++it) {
        result += it.value();  
        }  

        return result; 
    } 
  };

#endif

