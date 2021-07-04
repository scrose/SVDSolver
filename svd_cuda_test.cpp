/**
* **********************************************
* Singular Value Decomposition: Matrix Class (CUDA)
* **********************************************
 * CSC 586B - Spring 2020 - Project: Interim Report #3
 * Author: Spencer Rose
 * GPGPU-enabled (CUDA) SVD solver
 *
* Data Structures:
*  - Matrix(): GPU Matrix
*  - Slice{}: stores matrix indices for a slice
* **********************************************
**/

#include <iostream>
#include <iomanip>
#include <cassert>
#include <typeinfo>
#include "timing.h"
#include "matrix_gpu.h"    // matrix class with operators
#include "svd_cpu.h"       // CPU equivalent functions
#include "timing.h"


/**
 * Generates n x [nrows x ncols matrix] of random values that range from min_val to max_val
 * [Adapted from CSC586C (Spring 2020) tutorial code]
 */
struct matrix_generator {
    // Parameters for random matrices
    typedef float T;
    size_t const nrows, ncols, n_;
    T const min_val, max_val;

    std::vector<csc586::gpu::Matrix<T>> operator()() const {
        std::vector<csc586::gpu::Matrix<T>> matrix_array;
        for (auto i = 0u; i < n_; ++i) {
            auto mat = csc586::gpu::Matrix<T>(nrows, ncols);
            mat.fill(min_val, max_val);
            matrix_array.push_back(mat);
        }
        return matrix_array;
    }
};



/**
 * Output command-line options
 */
void print_help() {
    std::cout << "Options for CUDA Testing" << std::endl;
    std::cout << "\n(1) Run benchmark tests for CUDA band reduction." << std::endl;
    std::cout << "\t>> benchmark [<int> Band size ] [<int> Step size] [<int> Number of steps]";
    std::cout << "[<int> Number of test instances]" << std::endl;
    std::cout << "\tExample: ./svd_cuda benchmark 20 200 16 20" << std::endl;
    std::cout << "\n(2) Correctness Test: Compares test matrix and corresponding band and bidiagonal reductions" << std::endl;
    std::cout << "\t>> check [64|512|1024 Row/Column sizes]" << std::endl;
    std::cout << "\tExample: ./svd_cuda check 64\n" << std::endl;
}

/*
 * ************************************************
 * Main Test Routine
 * ************************************************
 */

int main(int argc, char *argv[]) {

    /*
     * ************************************************
     * Parameter and Matrix Initialization
     * ************************************************
     */

    //auto const num_trials = 20000u;

    // Initialize test input matrices
    csc586::gpu::Matrix<float> A;
    csc586::gpu::Matrix<float> B;
    csc586::gpu::Matrix<float> C;
    csc586::gpu::Matrix<float> band_check;
    csc586::gpu::Matrix<float> brd_check;


    // Initialize timing parameters
    auto start_time = std::chrono::system_clock::now();
    auto end_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (argc == 2 && strncmp(argv[1], "test", 4) == 0) {
        A = csc586::gpu::Matrix<float>(960, 960);
        A.fill(0, 5);
        csc586::gpu::brd_p1(A, 32u);
        A.print(16);
        exit(0);
    }

    // User input arguments: matrix size as rows x cols
    if (argc >= 2) {

        /*
         * ************************************************
         * GPU Band Reduction (Correctness Check)
         * ************************************************
         * Check correctness of Band Reduction against baseline
         */
        if (strncmp(argv[1], "check", 5) == 0) {
            std::cout << "Checking correctness ... " << std::endl;
            size_t size = size_t(atoi(argv[2]));
            // Fixed band size
            size_t const band_size = 4u;

            // Read test matrix
            A = csc586::gpu::Matrix<float>(size, size);
            std::string filename =
                    std::string("../data/test_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
                    std::string(".bin");
            std::cout << "Reading file: " << filename << std::endl;
            A.read(filename);
            A.print();

            // Run CUDA band reduction
            std::cout << "\n\nCUDA Test (Band):" << std::endl;
            csc586::gpu::brd_p1(A, band_size);
            A.print(16);

            // Compare with Baseline results
            std::cout << "\n\nBaseline Test (Band):" << std::endl;
            filename =
                    std::string("../data/band_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
                    std::string(".bin");
            band_check = csc586::gpu::Matrix<float>(size, size);
            band_check.read(filename);
            band_check.print(16);

            // Calculate Error
            auto error = A.mse(band_check, band_size);
            std::cout << "\n\nMSE of Band Reduction: " << error << std::endl;


            // Run CUDA bidiagonal reduction
            std::cout << "\n\nCUDA Test (Bidiagonal):" << std::endl;
            csc586::gpu::brd(A);
            A.print(10);

            // Compare with Baseline results
            std::cout << "\n\nBaseline Test (Bidiagonal):" << std::endl;
            filename =
                    std::string("../data/bidiagonal_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
                    std::string(".bin");
            band_check = csc586::gpu::Matrix<float>(size, size);
            band_check.read(filename);
            band_check.print(10);

            // Calculate Error
            error = A.mse(band_check, 2);
            std::cout << "\n\nMSE of Bidiagonal Reduction: " << error << std::endl;
        }


        /*
         * ************************************************
         * Time Band or Bidiagonal Reduction against baseline
         * ************************************************
         */

        // Run baseline benchmark for given user input parameters
        else if ( (argc > 5) && (strncmp(argv[1], "benchmark", 9) == 0) ) {

            // initialize benchmark parameters
            typedef float T;
            T const min_val = 0;
            T const max_val = 5;

            // Size of band (tile width)
            auto b_size = size_t(atoi(argv[2]));
            // Step in size of matrix for each iteration
            auto step = size_t(atoi(argv[3]));
            // Number of steps
            auto n = size_t(atoi(argv[4]) + 1);
            // Number of test instances for benchmark
            auto n_test_instances = size_t(atoi(argv[5]));
            // Results array to write to file
            //std::ostringstream vts;
            std::vector<int> x;
            std::vector<float> y;

            std::cout << "Benchmark: CUDA Band Reduction" << std::endl;
            std::cout << "\tBand size: " << b_size << std::endl;
            std::cout << "\tStep size: " << step << std::endl;
            std::cout << "\tNumber of steps: " << n - 1 << std::endl;
            std::cout << "\tNumber of test instances: " << n_test_instances << std::endl;

            // Seed for the random number generator (current time)
            std::srand(static_cast< uint32_t >( std::time(0)));

            // Function references
            csc586::gpu::Matrix<T> (*brd_p1)(csc586::gpu::Matrix<T> &, const size_t) = csc586::gpu::brd_p1;

            // Run diagnostic loop for matrix size N = k * step
            std::cout << "Average time per CUDA Band Reduction" << std::endl;
            for (auto k = 1u; k < n; ++k) {

                // Parameters for random value matrices
                size_t const rows = k * step;
                size_t const cols = k * step;

                // Run the benchmark on each algorithm/implementation, recording the average time taken.
                auto gen1 = matrix_generator{rows, cols, n_test_instances, min_val, max_val};
                std::vector<csc586::gpu::Matrix<T> > matrix_data = gen1();
                auto const avg_time = csc586::benchmark::benchmark(brd_p1, matrix_data, b_size);
                std::cout << "N = " << cols << " | " << avg_time*1e-6 << " sec" << std::endl;

                // Update results array
                x.push_back(static_cast< int >(k * step));
                y.push_back(static_cast< float >(avg_time*1e-6));

            std::cout << "Done." << std::endl;
        }
        // Provided arguments are not defined.
        else {
            print_help();
        }
    }
    // Provided arguments are not defined.
    else {
        print_help();
    }


    return 0;

}

