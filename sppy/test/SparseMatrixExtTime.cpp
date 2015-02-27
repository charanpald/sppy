#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <time.h> 

using namespace Eigen;

std::vector<long> get_vector_of_rn_longs(long length, long min, long max, int seed);
std::vector<double> get_vector_of_rn_doubles(int length, double min, double max);

int main()
{
  long N_COLS = 10000;
  long N_ROWS = 10000;
  long N_VALUES = N_COLS*N_ROWS*0.01;

  std::cout << N_VALUES << std::endl;

  SparseMatrix<double, RowMajor> big_A(N_ROWS, N_COLS);
  std::vector<long> cols_a = get_vector_of_rn_longs(N_VALUES, 0, N_COLS, 342);
  std::vector<long> rows_a = get_vector_of_rn_longs(N_VALUES, 0, N_ROWS, 43);
  std::vector<double> values_a = get_vector_of_rn_doubles(N_VALUES, 0, 1);

  for (int i = 0; i < N_VALUES; i++) {
    big_A.coeffRef(rows_a[i], cols_a[i]) = values_a[i];
    } 

  std::cout << big_A.nonZeros() << "  " << big_A.rows() << std::endl;
  big_A.makeCompressed();

  SparseMatrix<double, ColMajor> big_B(N_ROWS, N_COLS);
  big_B = big_A.transpose();
  big_B.makeCompressed();

  SparseMatrix<double, RowMajor> big_AB(N_ROWS, N_COLS);
  big_AB.reserve(58949434);

  clock_t begin = clock();
  //(big_A * big_B);
  for (int i = 0; i < 3; i++) 
      big_AB = (big_A * big_B); //.pruned();
  
  std::cout << big_AB.nonZeros() << "  " << big_AB.rows() << std::endl;

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / (CLOCKS_PER_SEC*3);
  std::cout << "Time taken : " << elapsed_secs << std::endl;
}

std::vector<long> get_vector_of_rn_longs(long length, long min, long max, int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_int_distribution<long> distribution(min, max-1);
    std::vector<long> my_vector(length);

    auto generator = std::bind(distribution, engine);
    std::generate_n(my_vector.begin(), length, generator); 

  return my_vector;
}

std::vector<double> get_vector_of_rn_doubles(int length, double min, double max)
{
    std::uniform_real_distribution<double> unif(min, max);
    std::default_random_engine re;
    std::vector<double> my_vector(length);

    for (int i=0; i < length; i++) {
        my_vector[i] = unif(re);
        }
  return my_vector;
}