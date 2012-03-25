#include "mapped_matrix_ext.h"
#include <boost/numeric/ublas/io.hpp>

int main () {
	using namespace boost::numeric::ublas;
	mapped_matrix_ext<double> m (3, 3, 3 * 3);
	for (unsigned i = 0; i < m.size1 (); ++ i)
                	for (unsigned j = 0; j < m.size2 (); ++ j)
            	    m (i, j) = 3 * i + j;
        
        	mapped_matrix_ext<double> m2 (3, 3, 3 * 3);
	for (unsigned i = 0; i < m2.size1 (); ++ i)
                	for (unsigned j = 0; j < m2.size2 (); ++ j)
            	    m2(i, j) = 3 * i + j;

            std::cout << m << std::endl;
            std::cout << m2 << std::endl;

            std::cout << m + m2 << std::endl;
            std::cout << m.add(m2) << std::endl; 

	}

