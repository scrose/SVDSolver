
/**
* **********************************************
* Singular Value Decomposition (Single-core Model)
 * **********************************************
 * CSC 586B - Spring 2020 - Final Project
 * Author: Spencer Rose
* **********************************************
 * Applies two-step SVD reduction of mxn matrix A
 * tto the form A = U\SigmaV^T where the columns
 * of U form an nxn orthonormal matrix; the rows
 * of V^T form an nxn orthonormal matrix, and \Sigma
 * is an m×n diagonal matrix with positive real
 * entries known as the singular values of A.
* Input:
*  - d: diagonal of B
*  - e: superdiagonal of B
* Functions:
*  - bidiag_reduce()
*  - diag_reduce()
*  - bidiag_block_reduce()
* **********************************************
**/

/**
 * Adapted from CSC586C (Sean Chester) benchmarking application
 */

#include <iostream> // for outputting (printing) to streams (e.g., the terminal)
#include <random> 	// std::rand, std::srand, std::default_random_engine
#include <fstream>  // file stream
#include <iomanip>
#include <string>
#include <sstream>
#include <omp.h>

#include "timing.h"       // Timing utilites
#include "matrix.h"       // Matrix class with operators
#include "svd_serial.h"   // Single-core SVD functions
#include "svd_parallel.h" // Multi-core SVD functions
#include <cstring>



/**
 * Generates n x [nrows x ncols matrix] of random values that range from min_val to max_val
 * [Adapted from CSC586C (Spring 2020) tutorial code]
 */
//template < typename T >
struct matrix_generator {
    // Parameters for random matrices
    size_t const nrows, ncols, n_;
    float const min_val, max_val;

    std::vector<csc586::Matrix<float>> operator()() const {
        std::vector<csc586::Matrix<float>> matrix_array;
        for (auto i = 0u; i < n_; ++i) {
            auto mat = csc586::Matrix<float>(nrows, ncols);
            mat.fill(min_val, max_val);
            matrix_array.push_back(mat);
        }
        return matrix_array;
    }
};

/**
* Generates n x [k and k-1 bidiagonal matrix] of random values that range from min_val to max_val
* [Adapted from CSC586C (Spring 2020) tutorial code]
*/
//template < typename T >
struct bidiagonal_generator {
    // Parameters for random matrices
    typedef float T;
    size_t const k, n_;
    T const min_val, max_val;

    std::vector<csc586::serial::Bidiagonal<T>> operator()() const {
        std::vector<csc586::serial::Bidiagonal<T>> bidiagonal_array;
        for (auto i = 0u; i < n_; ++i) {
            auto mat = csc586::Matrix<T>(2, k);
            mat.fill(min_val, max_val);
            auto d = mat[0];
            auto e = mat[1];
            e.resize(k-1);
            csc586::serial::Bidiagonal<T> B = {d,e};
            bidiagonal_array.push_back(B);
        }
        return bidiagonal_array;
    }
};


/**
 * Output command-line options
 */
void print_help() {
        std::cout << "SVD (CPU) Benchmarking Tool" << std::endl;

        std::cout << "\nDescription: Executes SVD benchmark tests for given computational model." << std::endl;
        std::cout << "\tOPTIONS: [base|singlecore|multicore|diagonal] [<int> Step size] [<int> Number of steps] [<int> Number of test instances] [<int> Block size ]";
        std::cout << "\n\tEXAMPLE: ./benchmark multicore 320 10 4 32" << std::endl;
        std::cout << "\nModel Options:" << std::endl;
        std::cout << "\tbase : Golub-Kahan Bidiagonal Reduction" << std::endl;
        std::cout << "\tsinglecore : Blocked (Panel) Bidiagonal Reduction (Requires Block Size)" << std::endl;
        std::cout << "\tmulticore : Tiled Bidiagonal Reduction (Requires Block Size)" << std::endl;
        std::cout << "\tdiagonal : QR Diagonalization" << std::endl;

}


/**
 * Runs benchmarks for SVD decomposition
 */
int main( int argc, char *argv[] ) {

    // Run benchmark for given user input parameters
    if ( argc > 4 ) {

        // initialize benchmark parameters
        typedef float T;
        T const min_val = 0;
        T const max_val = 5;

        // Model option value
        auto model_option = argv[1];
        // Step in size of matrix for each iteration
        auto step = size_t(atoi(argv[2]));
        // Number of steps
        auto n = size_t(atoi(argv[3]) + 1);
        // Number of test instances for benchmark
        auto n_test_instances = size_t(atoi(argv[4]));
        // Block size
        size_t b_size = 0u;

        // model ID and label
        int model = 0;
        std::string name = "";


        std::vector<std::string> results;

        // Select model to benchmark
        if (strncmp(model_option, "base", 4) == 0) {
            model = 1;
            name = "Golub-Kahan Bidiagonal Reduction (Single-core)";
        }
        else if (strncmp(model_option, "singlecore", 10) == 0) {
            model = 2;
            name = "Blocked (Panel) Bidiagonal Reduction (Single-core)";
        }
        else if (strncmp(model_option, "multicore", 8) == 0) {
            model = 3;
            name = "Tiled Bidiagonal Reduction (Multi-core)";
        }
        else if (strncmp(model_option, "diagonal", 8) == 0) {
            model = 4;
            name = "QR Diagonalization Reduction (Single-core)";
        }
        else {
            print_help();
            exit(0);
        }

        // Output to console
        std::cout << "\nBenchmark: " << name << std::endl;

        // Size of band (tile width)

        if ( model == 2 || model == 3 ) {
            if ( argc < 6 ) {
                std::cout << "\nError: Block size is required.\n" << std::endl;
                print_help();
                exit(0);
            }
            b_size = size_t(atoi(argv[5]));
            std::cout << "\tBlock size: " << b_size << std::endl;
        }

        std::cout << "\tStep size: " << step << std::endl;
        std::cout << "\tNumber of steps: " << n - 1 << std::endl;
        std::cout << "\tNumber of test instances: " << n_test_instances << std::endl;

        // Seed for the random number generator (current time)
        std::srand(static_cast< uint32_t >( std::time(0)));

        // Results array to write to file
        std::ostringstream vts;
        std::vector<int> x;
        std::vector<float> y;
        std::vector<float> z;

        // Function references
        // [1] Golub-Kahan Bidiagonal Reduction
        csc586::serial::Bidiagonal<T>(*brd)( csc586::Matrix<T> & ) = csc586::serial::brd;
        // [2] Blocked (Panel) Bidiagonal
        csc586::serial::Bidiagonal<T> (*block_brd)( csc586::Matrix<T> &, const size_t ) = csc586::serial::block_brd;
        // [3a] Tiled Dense-to-Band Reduction
        csc586::Matrix<T> (*brd_p1)(csc586::Matrix<T> &, const size_t) = csc586::parallel::brd_p1;
        // [3b] Tiled Dense-to-Band Reduction
        csc586::serial::Bidiagonal<T> (*brd_p2)(csc586::Matrix<T> &, const size_t) = csc586::parallel::brd_p2;
        // [4] QR Diagonalization
        csc586::serial::Bidiagonal<T> (*qrd)(csc586::serial::Bidiagonal<T> &) = csc586::serial::qrd;

        std::cout << "\nAverage time per " << name << ":" << std::endl;

        // Run diagnostic loop for matrix size N = k * step
        for (auto k = 1u; k < n; ++k) {

            // Parameters for random value matrices
            size_t const rows = k * step;
            size_t const cols = k * step;

            // Timings
            double avg_time_1 = 0;
            double avg_time_2 = 0;

            auto gen1 = matrix_generator{rows, cols, n_test_instances, min_val, max_val};
            auto gen2 = bidiagonal_generator{cols, n_test_instances, min_val, max_val};

            std::vector<csc586::Matrix<T> > matrix_data = gen1();
            std::vector<csc586::serial::Bidiagonal<T> > bidiagonal_data = gen2();

            // Print matrix size
            std::cout << "\n\tN = " << cols << " : ";

            // Select Model
            switch (model) {
                case 1 :    avg_time_1 = csc586::benchmark::benchmark(brd, matrix_data) * 1e-6;
                            std::cout <<  avg_time_1 << " sec (dense -> bidiagonal)";
                            break;
                case 2 :    avg_time_1 = csc586::benchmark::benchmark(block_brd, matrix_data, b_size) * 1e-6;
                            std::cout <<  avg_time_1 << " sec (dense -> bidiagonal)";
                            break;
                case 3 :    avg_time_1 = csc586::benchmark::benchmark(brd_p1, matrix_data, b_size) * 1e-6;
                            std::cout <<  avg_time_1 << " sec (dense -> band) | ";
                            avg_time_2 = csc586::benchmark::benchmark(brd_p2, matrix_data, b_size) * 1e-6;
                            std::cout <<  avg_time_2 << " sec (band -> bidiagonal) | ";
                            std::cout <<  avg_time_1 + avg_time_2 << " sec (total)";
                            break;
                case 4 :    avg_time_1 = csc586::benchmark::benchmark(qrd, bidiagonal_data) * 1e-6;
                            std::cout <<  avg_time_1 << " sec (bidiagonal -> diagonal) | ";
                            break;
                default :   print_help();
                            exit(0);
            }

            // Update results array
            x.push_back(static_cast< int >(k * step));
            y.push_back(static_cast< float >(avg_time_1 * 1e-6));
            if ( model == 3 )
                z.push_back(static_cast< float >(avg_time_2 * 1e-6));

        }

        // Write benchmark results to file
        if (!x.empty() && !y.empty()) {
            // Convert all but the last element to avoid a trailing ","
            std::copy(x.begin(), x.end() - 1,
                      std::ostream_iterator<int>(vts, ", "));

            // Now add the last element with no delimiter
            vts << x.back();
            vts << "\n";

            std::copy(y.begin(), y.end() - 1,
                      std::ostream_iterator<float>(vts, ", "));

            // Now add the last element with no delimiter
            vts << y.back();

            // Include extra timings if provided
            if ( model == 3 ) {
                std::copy(z.begin(), z.end() - 1,
                          std::ostream_iterator<float>(vts, ", "));

                // Now add the last element with no delimiter
                vts << z.back();
            }
        }

        // Write data stream to file
        auto filename = std::string("data/" + std::string(model_option) + "_benchmark.csv");
        std::cout << "\n\nWriting results to file " << filename << " ... ";
        std::ofstream ftest;
        ftest.open(filename);
        ftest << vts.str();
        ftest.close();
        std::cout << "done." << std::endl;

    }
    else {
        print_help();
    }

    return 0;

}
