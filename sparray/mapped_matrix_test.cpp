#include "mapped_matrix_ext.h"
#include <boost/numeric/ublas/io.hpp>

int main () {
	using namespace boost::numeric::ublas;
         using namespace std;
	mapped_matrix_ext<double> m (3, 3, 3 * 3);
	for (unsigned i = 0; i < m.size1 (); ++ i)
                	for (unsigned j = 0; j < m.size2 (); ++ j)
            	    m (i, j) = 3 * i + j;
        
        	mapped_matrix_ext<double> m2 (3, 3, 3 * 3);
	for (unsigned i = 0; i < m2.size1 (); ++ i)
                	for (unsigned j = 0; j < m2.size2 (); ++ j)
            	    m2(i, j) = 2 * i + j;

    mapped_matrix_ext<double> n, n3; 
    mapped_matrix_ext<double> *n2; 
    cout << m << endl; 
    cout << m2 << endl; 
    cout << m + m2 << endl; 
    cout << m.add(m2) << endl; 


    n = m.add(m2);
    cout << n.multiply(2) << endl; 

    n2 = &n; 
    cout << *n2 << endl; 
        //double c = norm_frobenius(m);
        //cout << c << endl; 

        //std::cout << m << std::endl;
        //std::cout << m2 << std::endl;

        //std::cout << m + m2 << std::endl;
        //std::cout << m.add(m2) << std::endl; 

	}

