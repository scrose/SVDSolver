/**
 * Unit Tests (CUDA)
 * **************************
 * NOTE: This file generates a unit test suite using the Catch2 header-only library.
 * The library contains a main method to make this a stand alone application,
 * which consists of all our tests.
 * **************************
 */

// The following two verbatim lines are all that are required to use Catch2.
// Note that you may have to adjust the path to catch.hpp in your include statement,
// depending on where you downloaded it to and how you have set your $PATH variables.

#define CATCH_CONFIG_MAIN       // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_ENABLE_BENCHMARKING
// This includes the catch header, which defines the entire library
#include "catch.hpp"

#include "matrix.h"  // test code
#include "svd_gpu.h"  // test code
#include "svd_cuda.h"  // test code


// Initialization
float const min_val = 0;
float const max_val = 5;
auto A = csc586::gpu::Matrix<float>(512, 512);
auto B = csc586::gpu::Matrix<float>(512, 20);
auto C = csc586::gpu::Matrix<float>(20, 20);
auto D = csc586::gpu::Matrix<float>(512, 20);
A.fill(min_val, max_val);
B.fill(1, 1);
C.fill(5, 5);
A.print();
B.print();

// GPU buffer Allocation
auto const n_partitions = 6;
auto const partition_dim = std::max(A.nrows, A.ncols);
auto const partition_size = partition_dim * partition_dim;
auto const buffer_size = sizeof(float) * n_partitions * partition_size;

// Initialize initial buffer pointer and allocate buffer memory on GPU
float *dev_init_ptr;
cudaMalloc((void **) &dev_init_ptr, buffer_size );

fprintf(stderr,"GPUassert0: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__);
fprintf(stderr,"GPUassert0: %s %s %d\n", cudaGetErrorString(cudaDeviceSynchronize()), __FILE__, __LINE__);

// define number of blocks per grid
csc586::gpu::dimGrid.x = static_cast<int> (ceil(float( A.ncols + csc586::gpu::dimBlock.x - 1 )/float( csc586::gpu::dimBlock.x ) ) );
csc586::gpu::dimGrid.y = static_cast<int> (ceil(float( A.nrows + csc586::gpu::dimBlock.y - 1 )/float( csc586::gpu::dimBlock.y ) ) );

csc586::gpu::MatrixCUDA dev_A = {A.nrows, A.ncols, dev_init_ptr};
csc586::gpu::MatrixCUDA dev_B = {B.nrows, B.ncols, dev_init_ptr + A.size()};
csc586::gpu::MatrixCUDA dev_C = {C.nrows, C.ncols, dev_B.elements + B.size()};
csc586::gpu::MatrixCUDA dev_D = {D.nrows, D.ncols, dev_C.elements + C.size()};

auto buffer = dev_C.elements + C.size();

csc586::gpu::copy( A, dev_A );
csc586::gpu::copy( B, dev_B );
csc586::gpu::copy( C, dev_C );
csc586::gpu::copy( D, dev_D );


// CUDA test scenarios

TEST_CASE( "Set Element Kernel", "[set_elem]" )
{
    set_elem( dev_A, 6, 3, -999 );
    copy(A, dev_A);
    // element is set
    REQUIRE( A[6][3] == -999 );
}


TEST_CASE( "Slice Kernel", "[slice]" ) {

    auto dev_slice1 = csc586::gpu::slice(dev_A, {5, 10, 6, 12}, buffer );
    auto slice1 = Matrix<float>( 5, 6 );
    copy(slice1, dev_slice1);
    auto slice2 = A.slice({5, 10, 6, 12})

    REQUIRE( slice1 == slice2 );
}



TEST_CASE("Matrix Multiplication Kernel", "[matmul]" ) {

    auto J = csc586::gpu::Matrix<float>(32, 15);
    auto K = csc586::gpu::Matrix<float>(15, 32);
    J.fill(1, 2);
    K.fill(5, 6);
    csc586::gpu::MatrixCUDA dev_J = {J.nrows, J.ncols, dev_init_ptr};
    csc586::gpu::MatrixCUDA dev_K = {K.nrows, K.ncols, dev_init_ptr + J.size()};
    csc586::gpu::copy( J, dev_J );
    csc586::gpu::copy( K, dev_K );

    auto X = J.mm(K);
    auto dev_X = csc586::gpu::matmul( dev_J, dev_K, buffer );
    copy(X2, dev_X);

    REQUIRE( X == X2 );

}


TEST_CASE("Copy Kernel", "[copy]" ) {

    csc586::gpu::copy(slice, dev_B, {5, 10, 6, 12} );

    REQUIRE( X == X2 );

}


TEST_CASE("Multiplication Kernel - Square", "[mm_square]" ) {

    J = csc586::gpu::Matrix<float>(20, 20);
    K = csc586::gpu::Matrix<float>(20, 32);
    J.fill(1, 2);
    K.fill(5, 6);
    dev_J = {J.nrows, J.ncols, dev_init_ptr};
    dev_K = {K.nrows, K.ncols, dev_init_ptr + J.size()};
    csc586::gpu::copy( J, dev_J );
    csc586::gpu::copy( K, dev_K );

    auto X2 = J.mm(K);
    auto X1 = csc586::Matrix<float>(20,32);
    dev_X = csc586::gpu::matmul( dev_J, dev_K, buffer );
    copy( X1, dev_X );

    REQUIRE( X1 == X2 );

}

TEST_CASE("Multiplication-PLUS+ Kernel", "[mm_plus]" ) {

    auto X1 = csc586::Matrix<float>(20,32);
    dev_X = csc586::gpu::matmul(dev_J, dev_K, buffer, 0, 1 );
    copy( X1, dev_X );
    K += J.mm(K);

    auto X2 = K;

    REQUIRE( X1 == X2 );


}

TEST_CASE("Transposition Kernel", "[transpose]" ) {

    auto X1 = D.transpose()
    csc586::gpu::transpose(dev_D, buffer);
    auto X2 = csc586::Matrix<float>(D.ncols, D.nrows);
    copy(X2, dev_D);

    REQUIRE( X1 == X2 );

}


TEST_CASE("Addition Kernel", "[add]" ) {

    // CPU
    auto F = D;
    F *= -5;
    F += B;
    X1 = F;

    // CUDA
    csc586::gpu::copy( B, dev_B );
    csc586::gpu::copy( D, dev_D );
    auto dev_G = csc586::gpu::add(dev_B, dev_D, buffer, -5.);
    auto X2 = csc586::Matrix<float>(B.nrows, B.ncols);
    copy(X2, dev_G);

    REQUIRE( X1 == X2 );

}



cudaFree(dev_init_ptr);
exit(0);

