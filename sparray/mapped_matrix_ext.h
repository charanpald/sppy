
#ifndef DATE_H
#define DATE_H
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/range/size_type.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;


template <class T>
class mapped_matrix_ext:public mapped_matrix<T> {
  public:
	mapped_matrix_ext<T>(int size1, int size2, int non_zeros=0): 
		mapped_matrix<T>(size1, size2, non_zeros){ 
		} 

	float get_item(int i, int j) const { 
		return (*this)(i, j); 
	}

	void set_item(int i, int j, T val) {
		(*this)(i, j) = val; 
		} 

        mapped_matrix_ext<T>& add(mapped_matrix_ext<T>& matrix_A)  { 
                //return (*this) + A;    
                (*this) += matrix_A;
                return *this;            
                }
  };

#endif

