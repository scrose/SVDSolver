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
#include "matrix_gpu.h"  // matrix class with operators
#include "svd_cpu.h"     // CPU equivalent functions
#include "timing.h"

#define cuda_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

// GPU error checking
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace csc586 { // anonymous
    namespace gpu {

        // Data type for indexing matrix slices
        struct MatrixCUDA {
            size_t nrows; // matrix rows
            size_t ncols; // matrix columns
            float *elements; // data buffer
            // word size of matrix
            size_t size() {
                return nrows * ncols;
            }

            // byte size of matrix
            int alloc() {
                return sizeof(float) * nrows * ncols;
            }

            // Set dimensions of 2D matrix
            void set_dim(size_t const rows, size_t const cols) {
                nrows = rows;
                ncols = cols;
            }

            // Print CUDA matrix data to std::out
            void print() {
                auto A_cpu = Matrix<float>(1, size());
                cudaMemcpy(&A_cpu[0][0], elements, alloc(), cudaMemcpyDeviceToHost);
                A_cpu = A_cpu.reshape(nrows, ncols);
                A_cpu.print();
            }
        };

        // Data type for CUDA-enabled Householder reflector
        // as projection vector w and scalar tau
        struct ReflectionCUDA {
            MatrixCUDA w; // Householder vector
            MatrixCUDA w_T; // Householder vector (transposed)
            MatrixCUDA tau; // scalar normalizer
        };


        /*
         * ===============================================
         * Initialize CUDA Matrix
         * -----------------------------------------------
         * Creates new MatrixCUDA matrix
         *
         * Input:
         *  - <float> m: row dimension
         *  - <float> n: column
         *  - <float>* buffer: CUDA buffer allocation
         *  Output:
         *  - MatrixCUDA A: CUDA matrix struct
         * ===============================================
         */

        MatrixCUDA create_matrix(size_t const m, size_t const n, float *&buffer_ptr, bool zero_set = false ) {

            MatrixCUDA A = {m, n, buffer_ptr};

            // Set allocation to zero if requested
            if (zero_set)
                cudaMemset(buffer_ptr, 0, A.alloc());

            buffer_ptr += A.size();
            return A;
        }


        // Message handler
        void message( std::string message ) {
            std::cout << "\n\n================\n" << message << "\n================\n\n" << std::endl;
        }

        /*
         * ===============================================
         * Constants
         * ===============================================
         */

        // Max number of threads per thread block = 2048.
        size_t const tile_size = 32u;
        // GPU block dimension (x, y, z)
        dim3 const dimBlock( tile_size, tile_size, 1u );
        // Grid size
        dim3 dimGrid( 1, 1 );
        // GPU cut-off
        size_t min_width = 64u;

        /*
         * ===============================================
         * Initialize Grid for invocation
         * -----------------------------------------------
         * Defines grid dimensions
         *
         * Input:
         *  - <float> m: y dimension
         *  - <float> n: x dimension
         * ===============================================
         */
        void init_grid( size_t const m, size_t const n ) {

            // define number of blocks per grid
            dimGrid.x = static_cast<int> (ceil(float(n + dimBlock.x - 1) / float(dimBlock.x)));
            dimGrid.y = static_cast<int> (ceil(float(m + dimBlock.y - 1) / float(dimBlock.y)));
        }



        /*
        * ===============================================
        * Matrix Copy - CUDA Kernel
        * -----------------------------------------------
        * Copy matrix data slice: tgt <- src
        *
        * Input:
        *  - MatrixCUDA src (source m x n 1D-array)
        *  - MatrixCUDA tgt (target m x n 1D-array)
        *  - <Slice> range: Range dimensions
        * Output:
        *  - tgt is overwritten with src
        * ===============================================
        */
        __global__
        void static copy_kernel(MatrixCUDA src, MatrixCUDA tgt, Slice const src_range, Slice const tgt_range )
        {
            // read the matrix tile into shared memory
            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            if((global_x < (tgt_range.j2 - tgt_range.j1)) && (global_y < (tgt_range.i2 - tgt_range.i1)))
            {
                auto const idx_src = (global_y + src_range.i1) * src.ncols + (global_x + src_range.j1);
                auto const idx_tgt = (global_y + tgt_range.i1) * tgt.ncols + (global_x + tgt_range.j1);

                tgt.elements[idx_tgt] = src.elements[idx_src];
            }
        }


        /*
        * ===============================================
        * Matrix Copy - Set single value
        * -----------------------------------------------
        * Set value of single matrix data element
        *
        * Input:
        *  - MatrixCUDA arr (source m x n 1D-array)
        *  - <float> idx index of array element
        *  - <float>/<float>* value: new element value
        * Output:
        *  - updates array in-place
        * ===============================================
        */
        __global__
        void set_val_kernel(float *arr, size_t const idx, float const value ) {
            arr[idx] = value;
        }

        __global__
        void set_val_ptr_kernel(float *arr, size_t const idx, float* value) {
            arr[idx] = value[0];
        }



        /*
         * ===============================================
         * Householder - Householder parameters
         * -----------------------------------------------
         * Change value of single matrix data element
         * REFERENCE: NVIDIA Programming Guide (2017)
         *
         * Input:
         *  - <float>* w: householder vector
         *  - <float>* w_T: householder vector (transposed)
         *  - <float>* norm: inner product of householder vector
         *  - <float>* tau: scaling factor
         * Output:
         *  - updates values in-place
         * ===============================================
         */

        __global__
        void hh_kernel( float*w, float*w_T, size_t const m, size_t const n, float* tau, float* norm ) {

            // Compute Euclidean normalization of w

            // Get square root of dot product
            auto norm_x = sqrt(norm[0]);
            // Get sign of initial value
            auto w1 = w[0];
            auto sgn = -copysignf(1, w1);
            // Compute u1
            auto u1 = w1 - sgn * norm_x;
            // Compute tau
            tau[0] = -sgn * u1 / norm_x;
            // invert u1
            auto alpha = 1. / u1;

            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            auto const idx = global_y * n + global_x;

            if((global_x < n) && (global_y < m)) {
                w[idx] *= alpha; // Scale householder
                w_T[idx] *= alpha; // Scale householder
            }
            // Set first value to one
            if ( idx == 0 ) {
                w[0] = 1.;
                w_T[0] = 1.;
            }

        }


        /*
         * ===============================================
         * Matrix Multiplication - CUDA Kernel (global)
         * -----------------------------------------------
         * Evaluated matrix product:
         * - C <- gamma*(alpha*A + AB)
         * - C <- gamma*(beta*B + AB)
         * CUDA device kernel (a.k.a., function). It runs on the GPU.
         * The `__global__` keyword identifies it as a device kernel.
         * Each thread will execute this function independently.
         * Because it is a device kernel, the pointers passed as arguments
         * must be allocated *on the device.* The input size, n, however,
         * is passed by value, not by pointer, so a separate copy is
         * created and it can reflect a host variable.
         *
         * Input:
         *  - <float>* A (m x p 1D-array)
         *  - <float>* B (p x n 1D-array)
         *  - <float>* result (m x n 1D-array)
         *  - Matrix dimensions: m, n, p
         *  - <float> alpha (constant in operation  A <- alpha*A + AB
         *  - <float> beta (constant in operation   B <- beta*B + AB
         *  - <float> gamma (constant in operation  result <- gamma*(result)
         *  - <float>* zeta (pointer to constant in operation  result <- gamma*(result)
         * Output:
         *  - Result array (result) is overwritten with alpha*A + AB OR alpha*B + AB.
         * ===============================================
         */

        __global__
        void static mm_kernel(float const *A, float const *B, float *result, size_t const m, size_t const n, size_t const p,
                       float const alpha, float const beta, float const gamma) {

            // Shared memory to hold tiles
            // tile_size is length (and width) of a thread submatrix
            __shared__ float A_tile[tile_size][tile_size];
            __shared__ float B_tile[tile_size][tile_size];

            // tile row / column
            auto const local_x = threadIdx.x;
            auto const local_y = threadIdx.y;

            // matrix row / column
            auto const global_x = blockIdx.x * tile_size + local_x;
            auto const global_y = blockIdx.y * tile_size + local_y;

            // Initialise the accumulation register
            float acc = 0.0f;

            size_t const n_tiles = (tile_size + p - 1) / tile_size;

            // Iterate over tiles
            for ( auto t = 0u; t < n_tiles; t++ ) {

                const int t_x = tile_size * t + local_x;
                const int t_y = tile_size * t + local_y;

                // Load data into shared memory tile
                A_tile[local_y][local_x] = (t_x < p && global_y < m) ? A[global_y * p + t_x] : 0.0;
                B_tile[local_y][local_x] = (t_y < p && global_x < n) ? B[t_y * n + global_x] : 0.0;

                __syncthreads();

                for (int k = 0; k < tile_size; ++k)
                    acc += A_tile[local_y][k] * B_tile[k][local_x];

                __syncthreads();
            }

            // Copy element to result matrix
            if( global_y < m && global_x < n ) {
                // Compute index
                auto idx = (blockIdx.y * blockDim.y + threadIdx.y) * n + (blockIdx.x * blockDim.x + threadIdx.x);
                // A <- gamma*(A + alpha*Result)
                if (alpha) {
                    result[idx] = gamma * (acc + alpha * A[idx] );
                }
                    // A <- gamma*(B + beta*Result)
                else if (beta) {
                    result[idx] = gamma * (acc + beta * B[idx]);
                }
                    // A <- gamma*(Result)
                else {
                    result[idx] = gamma * acc;
                }
            }
            __syncthreads();
        }




        /*
         * ===============================================
         * Matrix Transpose - CUDA Kernel
         * -----------------------------------------------
         * Evaluated matrix product: A <- gamma(alpha*A + AB)
         * Uses shared memory tiles to coalesce global reads
         * and writes.
         *
         * Input:
         *  - <float>* A (m x n 1D-array)
         *  - <float>* A_T (n x m 1D-array)
         *  - Matrix dimensions: m, n
         * Output:
         *  - A_T is overwritten with A' (transpose)
         * ===============================================
         */
        __global__
        void static trans_kernel(float *A_T, float const *A, size_t const m, size_t const n )
        {
            __shared__ float block[tile_size][tile_size + 1];

            // read the matrix tile into shared memory
            // load one element per thread from device memory (idata) and store it
            // in transposed order in block[][]
            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            if((global_x < n) && (global_y < m))
            {
                auto const idx_in = global_y * n + global_x;
                block[threadIdx.y][threadIdx.x] = A[idx_in];
            }

            // synchronise to ensure all writes to block[][] have completed
            __syncthreads();

            // write the transposed matrix tile to global memory (odata) in linear order
            global_x = blockIdx.y * tile_size + threadIdx.x;
            global_y = blockIdx.x * tile_size + threadIdx.y;

            if((global_x < m) && (global_y < n))
            {
                auto const idx_out = global_y * m + global_x;
                A_T[idx_out] = block[threadIdx.x][threadIdx.y];
            }
            __syncthreads();
        }


        /*
         * ===============================================
         * Matrix Addition/Substraction - CUDA Kernel
         * -----------------------------------------------
         * Evaluated matrix difference: result <- A - alpha * B
         *
         * Input:
         *  - <float>* A (m x n 1D-array)
         *  - <float>* B (m x n 1D-array)
         *  - <float>* result (m x n 1D-array)
         *  - Matrix dimensions: m, n
         *  - <float> alpha (scaling factor)
         * Output:
         *  - results is overwritten with A - alpha * B
         * ===============================================
         */
        __global__
        void static add_kernel(float const *A, float const *B, float *result,
                               size_t const m, size_t const n, float const alpha )
        {
            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            auto const idx = global_y * n + global_x;

            if((global_x < n) && (global_y < m))
                result[idx] = A[idx] + alpha * B[idx];
        }

        /*
         * ===============================================
         * Matrix Scale - CUDA Kernel
         * -----------------------------------------------
         * Evaluated matrix difference: result <- A - alpha * B
         *
         * Input:
         *  - <float>* A (m x n 1D-array)
         *  - Matrix dimensions: m, n
         *  - <float> alpha (scaling factor)
         * Output:
         *  - results is overwritten with A <- alpha * A
         * ===============================================
         */
        __global__
        void static scale_kernel(float *A, size_t const m, size_t const n, float const alpha )
        {
            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            auto const idx = global_y * n + global_x;

            if((global_x < n) && (global_y < m))
                A[idx] *= alpha;
        }

        __global__
        void static scale_ptr_kernel(float *A, size_t const m, size_t const n, float* alpha )
        {
            auto global_x = blockIdx.x * tile_size + threadIdx.x;
            auto global_y = blockIdx.y * tile_size + threadIdx.y;

            auto const idx = global_y * n + global_x;

            if((global_x < n) && (global_y < m))
                A[idx] *= alpha[0];
        }



        /*
        * ===============================================
        * Matrix Copy (and variants) - Host Wrapper
        * -----------------------------------------------
        * Copy matrix data slice: tgt <- src
        *
        * Input:
        *  - MatrixCUDA src (source m x n 1D-array)
        *  - MatrixCUDA tgt (target m x n 1D-array)
        *  - <Slice> range: Range dimensions
        * Output:
        *  - tgt is overwritten with src
        * ===============================================
        */


        // [OVERLOADED] Copy Device matrix to Device matrix [source -> target]
        void copy( MatrixCUDA &src, MatrixCUDA &tgt, Slice const src_range, Slice const tgt_range ) {

            assert( (src_range.i2 - src_range.i1) == (tgt_range.i2 - tgt_range.i1) && "Slice rows are out of range.");
            assert( (src_range.j2 - src_range.j1) == (tgt_range.j2 - tgt_range.j1) && "Slice columns are out of range.");

            init_grid( src_range.i2 - src_range.i1, src_range.j2 - src_range.j1 );
            copy_kernel <<< dimGrid, dimBlock >>> ( src, tgt, src_range, tgt_range );

        }


        // [OVERLOADED] Copy Device matrix to Device matrix [full copy -> target]
        void copy( MatrixCUDA &src, MatrixCUDA &tgt, Slice const tgt_range ) {

            assert( src.nrows == (tgt_range.i2 - tgt_range.i1) && "Slice rows are out of range.");
            assert( src.ncols == (tgt_range.j2 - tgt_range.j1) && "Slice columns are out of range.");

            init_grid( tgt_range.i2 - tgt_range.i1, tgt_range.j2 - tgt_range.j1 );
            copy_kernel <<< dimGrid, dimBlock >>> ( src, tgt, {0, src.nrows, 0, src.ncols}, tgt_range );
        }


        // [OVERLOADED] Copy Device matrix to Device matrix [full copy -> full copy]
        void copy( MatrixCUDA &src, MatrixCUDA &tgt ) {

            assert(src.nrows == tgt.nrows && src.ncols == tgt.ncols && "Source and target copy is out of range.");

            cudaMemcpy(tgt.elements, src.elements, src.alloc(), cudaMemcpyDeviceToDevice);

        }


        // Returns slice copied from Device matrix to Device matrix
        MatrixCUDA slice( MatrixCUDA &src, Slice const range, float*& buffer) {

            assert(src.nrows >= (range.i2 - range.i1) && src.ncols >= (range.j2 - range.j1) && "Slice is out of range.");

            MatrixCUDA tgt = create_matrix( range.i2 - range.i1, range.j2 - range.j1, buffer );
            copy(src, tgt, range, {0, tgt.nrows, 0, tgt.ncols});

            // return slice
            return tgt;
        }


        // Reduces size of Device matrix
        void reduce( MatrixCUDA &src, Slice const range, float*& buffer ) {

            assert(src.nrows >= (range.i2 - range.i1) && src.ncols >= (range.j2 - range.j1) && "Slice is out of range.");

            MatrixCUDA tgt = create_matrix( range.i2 - range.i1, range.j2 - range.j1, buffer );
            copy(src, tgt, range, {0, tgt.nrows, 0, tgt.ncols});
            // Reset dimensions of source matrix
            src.set_dim( range.i2 - range.i1, range.j2 - range.j1 );
            // Copy src <- tgt
            copy(tgt, src);
            // Reset buffer
            buffer -= tgt.size();

        }

        // [OVERLOADED] Copy Host matrix to Device matrix
        void copy( Matrix<float>& A_cpu, MatrixCUDA& A_gpu ) {

            assert(A_cpu.nrows == A_gpu.nrows && A_cpu.ncols == A_gpu.ncols && "CPU matrix dimensions must match GPU dimensions.");

            // load matrices
            auto A_1d = A_cpu.flatten();
            cudaMemcpy(A_gpu.elements, &A_1d[0][0], sizeof(float) * A_cpu.size(), cudaMemcpyHostToDevice);
            A_cpu = A_1d.reshape(A_cpu.nrows, A_cpu.ncols);

        }


        // [OVERLOADED] Copy Device matrix to Host matrix
        void copy( MatrixCUDA& A_gpu, Matrix<float>& A_cpu ) {

            assert(A_cpu.nrows == A_gpu.nrows && A_cpu.ncols == A_gpu.ncols && "CPU matrix dimensions must match GPU dimensions.");

            // load matrices
            auto A_1d = A_cpu.flatten();
            cudaMemcpy(&A_1d[0][0], A_gpu.elements, sizeof(float) * A_gpu.size(), cudaMemcpyDeviceToHost);
            A_cpu = A_1d.reshape(A_cpu.nrows, A_cpu.ncols);
        }




        /*
         * ===============================================
         * Matrix Multiplication - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix product: C <- AB
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - MatrixCUDA A (m x p matrix)
         *  - MatrixCUDA B (p x n matrix)
         *  - MatrixCUDA C (m x n matrix)
         *  - <float> alpha (constant in operation: C <- alpha*A + AB
         *  - <float> beta (constant in operation: C <- beta*B + AB
         *  - <float> gamma (constant in operation: C <- zeta(AB)
         *  - <float> zeta (pointer to constant in operation: C <- zeta(AB)
         * Output:
         *  - Returns matrix C <- alpha*A + AB OR beta*B + AB.
         * ===============================================
         */

        MatrixCUDA matmul( MatrixCUDA A, MatrixCUDA B, float*& buffer, float const alpha = 0.0,
                float const beta = 0.0, float const gamma = 1.0 ) {

            assert(A.ncols == B.nrows && "Matrix 1 col dim must match Matrix 2 row dim.");
            assert( !(alpha > 0 && beta > 0) && "Matrix self-addition must be for matrix A or B, not both.");

            // Parameters
            auto m = A.nrows;
            auto n = B.ncols;
            auto p = A.ncols;

            // Initialize Grid
            init_grid( m, n );

            // Define resultant matrix parameters
            // Initialize transposed matrix
            MatrixCUDA C = create_matrix( m, n, buffer );

            // Matrix Multiplication kernel
            mm_kernel <<< dimGrid, dimBlock >>> (A.elements, B.elements, C.elements, m, n, p, alpha, beta, gamma );

            // return product
            return C;
        }



        /*
         * ===============================================
         * Matrix Transpose - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix transpose: A' <- transpose(A)
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - MatrixCUDA A  (m x n matrix)
         * Output:
         *  - MatrixCUDA A' (n x m matrix)
         * ===============================================
         */

        MatrixCUDA transpose( MatrixCUDA &A, float*& buffer ) {

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize transposed matrix
            MatrixCUDA A_T = create_matrix( n, m, buffer );

            // Initialize Grid
            init_grid( m, n );

            // Matrix transpose kernel
            trans_kernel <<< dimGrid, dimBlock >>> ( A_T.elements, A.elements, m, n );

            // return transposed matrix
            return A_T;
        }


        /*
         * ===============================================
         * Matrix Addition/Substraction - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix addition: C <- A + alpha * B
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - MatrixCUDA A (m x n matrix)
         *  - MatrixCUDA B (m x n matrix)
         * Output:
         *  - MatrixCUDA C (m x n matrix) = A + alpha * B
         * ===============================================
         */

        MatrixCUDA add( MatrixCUDA &A, MatrixCUDA &B, float*& buffer, float const alpha = 1.0 ) {

            assert(A.nrows == B.nrows && A.ncols == B.ncols && "Matrix dimensions must match for addition.");

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize resultant matrix
            MatrixCUDA C = create_matrix( m, n, buffer );

            // Initialize Grid
            init_grid( m, n );

            // Matrix Addition/Subtraction kernel
            add_kernel <<< dimGrid, dimBlock >>> (A.elements, B.elements, C.elements, m, n, alpha );

            // return sum
            return C;
        }

        /*
         * ===============================================
         * Matrix Scale - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix addition: C <- A + alpha * B
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - MatrixCUDA A (m x n matrix)
         *  - MatrixCUDA B (m x n matrix)
         * Output:
         *  - MatrixCUDA C (m x n matrix) = A + alpha * B
         * ===============================================
         */

        void scale( MatrixCUDA &A, float const alpha = 1.0 ) {

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize Grid
            init_grid( m, n );

            // Matrix Addition/Subtraction kernel
            scale_kernel <<< dimGrid, dimBlock >>> (A.elements, m, n, alpha );

        }

        void scale( MatrixCUDA &A, MatrixCUDA alpha ) {

            assert(alpha.nrows == 1 && alpha.ncols == 1 && "Scaling matrix must be 1 x 1.");

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize Grid
            init_grid( m, n );

            // Matrix Addition/Subtraction kernel
            scale_ptr_kernel <<< dimGrid, dimBlock >>> (A.elements, m, n, alpha.elements );

        }


        /*
      * ===============================================
      * Matrix Set Element - Host Wrapper
      * -----------------------------------------------
      * Sets value of matrix element A[i][j]
      * CPU host function that contains built-in CUDA
      * functions to initialize the GPU kernel.
      *
      * Input:
      *  - MatrixCUDA A (m x n matrix)
      *  - <size_t> i : row index
      *  - <size_t> j : column index
      *  - <float> value : value to set
      * Output:
      *  - A[i][j] is overwritten with value.
      * ===============================================
      */

        void set_elem( MatrixCUDA A, size_t const i, size_t const j, float const value ) {

            assert(i < A.nrows && j < A.ncols && "Index is out of range for matrix.");

            auto idx = A.nrows * i + j;

            // Set matrix value kernel
            set_val_kernel << < 1, 1 >> > (A.elements, idx, value );
        }


        void set_elem( MatrixCUDA A, size_t const i, size_t const j, MatrixCUDA value ) {

            assert(i < A.nrows && j < A.ncols && "Index is out of range for matrix.");

            auto idx = A.nrows * i + j;

            // Set value to matrix value
            cudaMemcpy(A.elements + idx, value.elements, sizeof(float), cudaMemcpyDeviceToDevice);
        }

        void get_elem( MatrixCUDA A, size_t const i, size_t const j, MatrixCUDA value ) {

            assert(i < A.nrows && j < A.ncols && "Index is out of range for matrix.");

            auto idx = A.nrows * i + j;

            // Set value to matrix value
            cudaMemcpy(value.elements, A.elements + idx, sizeof(float), cudaMemcpyDeviceToDevice);
        }




        /*
            * ===============================================
            * Householder Reflection - Host Wrapper
            * -----------------------------------------------
            * Input: Column vector w
            * Output: x (projection vector); tau (scaling factor)
            * ===============================================
            */
        ReflectionCUDA hholder_cuda( MatrixCUDA w, float*& buffer ) {

            auto m = w.nrows;
            auto n = w.ncols;

            assert( w.ncols == 1 && "Householder applied to column vectors only." );

            // Calculate matrices
            auto tau = create_matrix( 1, 1, buffer, true );
            auto w_T = transpose( w, buffer );
            auto norm = matmul( w_T, w, buffer );

            // Compute Householder reflector parameters
            init_grid( m, n );
            hh_kernel <<< dimGrid, dimBlock >>> ( w.elements, w_T.elements, m, n, tau.elements, norm.elements );

            return ReflectionCUDA{ w, w_T, tau };

        }




        /*
          * ===============================================
          * Compact YT represent. of Householder reflectors
          * Hk...H3.H2.H1 = I - VSV'
          * -----------------------------------------------
          * See Schreiber and Van Loan (1989)
          *
          * Input:
          *  - <float>* tau : Householder scalar factor
          *  - MatrixCUDA S: k x k Triangular matrix T
          *  - MatrixCUDA V: n x k Householder vectors Y
          * Output:
          *
          *  - S_k+1 = [ S_k    −tau_k+1.S_k.V_k'.v_k+1 ]
          *            [  0              -tau_k+1       ]
          * ===============================================
          */

        void wy_compact_cuda( const size_t j, MatrixCUDA tau, MatrixCUDA S, MatrixCUDA V, float*& buffer ) {

            // base case
            if ( j == 0u ) {
                //S[0][0] = -tau;
                scale(tau, -1.0f);
                set_elem( S, 0, 0, tau );
            }
            else {
                scale(tau, -1.0f);
                auto S_k = slice( S, {0, j, 0, j}, buffer );
                auto v = slice( V, {0, V.nrows, j, j+1}, buffer );
                auto V_k = slice( V, {0, V.nrows, 0, j}, buffer );
                auto V_k_T = transpose( V_k, buffer );
                auto z = matmul( V_k_T, v, buffer);
                z = matmul( S_k, z, buffer);
                scale( z, tau );
                // copy T_01 = z
                copy( z, S, {0, z.nrows, S_k.ncols, S_k.ncols + z.ncols});
                // copy T_11 = -tau
                set_elem( S, S_k.nrows, S_k.ncols, tau );
            }
        }


        /*
        * ===============================================
        * Panel QR Decomposition
        * -----------------------------------------------
        * Subroutine performs a panel QR factorization
        * of a matrix A of size m x n. This operation produces
        * an upper triangular matrix R, a unit lower triangular
        * matrix V that contains b Householder reflectors
        * and an upper triangular matrix Y as defined by the
        * compact WY technique.
         *
        * Input:
        *  - MatrixCUDA A: m x n panel matrix
        *  - MatrixCUDA S: m x m upper triangular matrix
        *  - MatrixCUDA V: m x n lower triangular matrix
        *  - <float>* buffer: CUDA buffer allocation
        * ===============================================
        */
        void qr_cuda( MatrixCUDA& A, MatrixCUDA& S, MatrixCUDA& V, float*& buffer_ptr ) {

            // Input matrix panel dimensions
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize buffer
            auto buffer = buffer_ptr;

            // Initialize container matrices
            auto R = create_matrix( m, n, buffer );
            auto Y = create_matrix( n, n, buffer, true );

            // Copy panel matrix data -> R
            copy(A, R);

            // Create reset point for buffer
            auto reset_buffer = buffer;

            // Reduction of rectangular matrix to upper triangular form
            for (auto j = 0u; j < std::min(n, m); ++j) {

                // compute Householder H_j from column A_j
                auto A_j = slice( R, {j, m, j, j + 1}, buffer );
                auto H_j = hholder_cuda( A_j, buffer );

                // Compute Y(i+1:n,i): y_i = tau * R'v
                auto A_trail = slice(R, {j, m, j, n}, buffer);
                A_trail = transpose( A_trail, buffer );

                auto y = matmul( A_trail, H_j.w, buffer );
                scale( y, H_j.tau );
                auto y_T = transpose( y, buffer );

                // Update matrices V, Y
                copy(H_j.w, V, {j, m, j, j + 1});
                copy(y_T, Y, {j, j + 1, j, n});

                // Store Housholder reflectors in WY compact form
                wy_compact_cuda( size_t(j), H_j.tau, S, V, buffer );

                // R <- A - VY
                auto VY = matmul( V, Y, buffer );
                R = add(A, VY, R.elements, -1. );

                // Reset buffer
                buffer = reset_buffer;

            }
            // update matrix with A <- A - VY
            copy( R, A );

            // Reset buffer
            buffer = buffer_ptr;
        }



        /*
        * ===============================================
        * Panel LQ Decomposition
        * -----------------------------------------------
        * Subroutine performs the panel LQ factorization
        * of a matrix A of size m x n. This operation produces
        * a lower triangular matrix R, a unit upper triangular
        * matrix U that contains b Householder reflectors
        * and a lower triangular matrix X as defined by the
        * compact WY technique.
        *
        * Input:
        *  - MatrixCUDA A: m x n panel matrix
        *  - MatrixCUDA S: m x m upper triangular matrix
        *  - MatrixCUDA V: m x n lower triangular matrix
        *  - <float>* buffer: CUDA buffer allocation
        * ===============================================
        */


        void lq_cuda( MatrixCUDA& A, MatrixCUDA& S, MatrixCUDA& U, float*& buffer_ptr ) {

            // Input matrix panel dimensions
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize buffer
            auto buffer = buffer_ptr;

            // Initialize container matrices
            auto L = create_matrix( m, n, buffer );
            auto X = create_matrix( m, m, buffer, true );

            // Copy trailing matrix data -> L
            copy( A, L );

            // Create reset point for buffer
            auto reset_buffer = buffer;

            // Reduction of rectangular matrix to lower triangular form
            for (auto i = 0u; i < std::min(n, m); ++i) {

                // Reduce row i of A = A - XU'
                // compute Householder H_i from column A_i
                auto A_i = slice( L, {i, i + 1, i, n}, buffer );
                auto A_i_T = transpose( A_i, buffer );
                auto H_i = hholder_cuda( A_i_T, buffer );

                // Compute X(i+1:n,i): x_i = tau * L'v
                auto A_trail = slice( L, {i, m, i, n}, buffer);
                auto x = matmul( A_trail, H_i.w, buffer );
                scale( x, H_i.tau );

                // Update matrices X, U
                copy(x, X, {i, m, i, i + 1});
                copy(H_i.w_T, U, {i, i + 1, i, n});
                auto U_T = transpose( U, buffer );

                // Store Housholder reflectors in WY compact form
                wy_compact_cuda( size_t(i), H_i.tau, S, U_T, buffer );

                // L <- A - XU'
                auto tmp = matmul( X, U, buffer );
                L = add( A, tmp, L.elements, -1. );

                // Reset buffer
                buffer = reset_buffer;

            }
            // update matrix with A <- A - VY
            copy( L, A );

            // Reset buffer
            buffer = buffer_ptr;

        }





        /*
         * ===============================================
         * QR Application - Host Wrapper
         * -----------------------------------------------
         * Computes A <- A(I + VSV') from compacted householders
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Subroutine applies S and V to matrix A
         * of a matrix A of size m x n for QR decomposition.
         *
         * Input:
         *  - MatrixCUDA A: m x n input matrix
         *  - MatrixCUDA S: m x m upper triangular matrix
         *  - MatrixCUDA V: m x n lower triangular matrix
         *  - <float>* buffer: CUDA buffer allocation
         * ===============================================
         */

        void qr_apply_cuda( MatrixCUDA &A, MatrixCUDA &S, MatrixCUDA &V, float*& buffer_ptr ) {

            // Initialize buffer address
            auto buffer = buffer_ptr;

            // Compute Q = I + VSV' from compacted householders
            auto V_T = transpose( V, buffer );
            auto SV_T = matmul( S, V_T, buffer );
            auto Q = matmul( V, SV_T, buffer );
            auto Q_T = transpose(Q, buffer );
            auto R = matmul( Q_T, A, buffer, 0, 1.);

            // Update A
            copy( R, A );

            // Reset buffer
            buffer = buffer_ptr;

        }




        /*
         * ===============================================
         * LQ Application - Host Wrapper
         * -----------------------------------------------
         * Computes A <- (I + VSV')'A from compacted householders
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Subroutine applies S and V to matrix A
         * of a matrix A of size m x n for LQ decomposition
         *
         * Input:
         *  - MatrixCUDA A: m x n input matrix
         *  - MatrixCUDA S: m x m upper triangular matrix
         *  - MatrixCUDA V: m x n lower triangular matrix
         *  - <float>* buffer: CUDA buffer allocation
         * ===============================================
         */

        void lq_apply_cuda( MatrixCUDA &A, MatrixCUDA &S, MatrixCUDA &U, float*& buffer_ptr ) {

            // Initialize buffer
            auto buffer = buffer_ptr;

            // Compute A <- A + A.(USU') from compacted householders
            auto U_T = transpose( U, buffer );
            auto SU = matmul( S, U, buffer );
            auto P = matmul( U_T, SU, buffer );
            auto L = matmul( A, P, buffer, 1, 0 );

            // Update A
            copy( L, A );

            // Reset buffer
            buffer = buffer_ptr;

        }



        /*
        * ===============================================
        * CUDA Blocked Band Reduction <cuda_brd_p1>
        * -----------------------------------------------
         * Dense matrix -> Banded matrix (Stage I of two-stage process)
         * -----------------------------------------------
         * Computes banded bidiagonal matrix B = U1'*A*V1 using
         * QR and LQ transformations to upper and lower diagonals
         * Input:
         *  - Matrix <float> A (m x n matrix)
         * Output:
         *  - Matrix <float> B (banded bidiagonal m x n matrix)
         * ===============================================
         */

        Matrix<float> cuda_brd_p1( Matrix <float>& A_cpu, size_t const b_size ) {

            // Input matrix dimensions
            auto const m = A_cpu.nrows;
            auto const n = A_cpu.ncols;

            init_grid(m, n);

            // GPU buffer allocation
            auto const n_partitions = 6;
            auto const partition_dim = std::max(A_cpu.nrows, A_cpu.ncols);
            auto const partition_size = partition_dim * partition_dim;
            auto const buffer_size = sizeof(float) * n_partitions * partition_size;

            // Initialize initial buffer pointer and allocate buffer memory on GPU
            float *buffer_ptr;
            cudaMalloc((void **) &buffer_ptr, buffer_size );

            // Initialize fixed container matrices
            auto A = create_matrix( m, n, buffer_ptr );
            auto A_trail = create_matrix( m, n, buffer_ptr );
            auto S = create_matrix( b_size, b_size, buffer_ptr );

            // Inittialize dynamic buffer pointer
            auto buffer = buffer_ptr;

            // Copy input matrix data to GPU
            copy( A_cpu, A );
            copy( A, A_trail );

            // Iterate over blocks of A (size: b_size)
            for ( auto k = 0u; k < n; k += b_size ) {

                // Cut-off for GPU
                // ---------------------------------
                if ( A_trail.size() <= min_width * min_width ) {

                    // Copy to CPU
                    copy(A, A_cpu);
                    auto A_trail_cpu = Matrix<float>(A_trail.nrows, A_trail.ncols);
                    copy(A_trail, A_trail_cpu);

                    // Reduce trailing matrix on CPU
                    A_trail_cpu = brd_p1( A_trail_cpu, b_size);

                    // A <- residual trailing matrix
                    A_cpu.copy( A_trail_cpu, { m - A_trail_cpu.nrows, m, n - A_trail_cpu.ncols, n } );
                    break;
                }
                // ---------------------------------

                // reset buffer
                buffer = buffer_ptr;

                // Extract QR panel
                auto panel_qr = slice( A_trail, { 0, A_trail.nrows, 0, b_size }, buffer );

                // Resize trailing matrix
                reduce( A_trail, {0, A_trail.nrows, b_size, A_trail.ncols}, buffer );

                // Initialize QR compact householder matrix
                auto V = create_matrix( panel_qr.nrows, b_size, buffer, true );

                // (Step 1) QR Reduction of left panel
                qr_cuda( panel_qr, S, V, buffer );

                // Update trailing matrix QA = (I − V T V')' A
                qr_apply_cuda( A_trail, S, V, buffer );

                // Copy QR reduced panel to A
                copy( panel_qr, A, {k, m, k, k + b_size});

                // LQ Panel Reduction
                if ( k + b_size < n - 1 ) {

                    // Extract LQ panel
                    auto panel_lq = slice( A_trail, {0, b_size, 0, A_trail.ncols}, buffer );

                    // Resize trailing matrix
                    reduce( A_trail, {b_size, A_trail.nrows, 0, A_trail.ncols}, buffer );

                    // Initialize LQ compact householder matrix
                    auto U = create_matrix( b_size, panel_lq.ncols, buffer, true );

                    // (Step 2) LQ Reduction of right panel
                    lq_cuda( panel_lq, S, U, buffer );

                    // Copy LQ reduced panel to A
                    copy( panel_lq, A, { k, k + b_size, k + b_size, n} );

                    // Update trailing matrix AQ' =  A (I − V T V')'
                    lq_apply_cuda( A_trail, S, U, buffer );

                }
                // A <- residual trailing matrix
                copy( A_trail, A, { m - A_trail.nrows, m, n - A_trail.ncols, n } );
            }

            // Free GPU allocation
            cudaFree(buffer_ptr);

            // Return band reduced matrix
            return A_cpu;
        }

    }  // GPU namespace
} // namespace csc586


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
    std::cout << "Options for CUDA-2 Testing" << std::endl;
    std::cout << "\n(1) Run benchmark tests for CUDA band reduction." << std::endl;
    std::cout << "\t>> benchmark [<int> Step size] [<int> Number of steps] [<int> Number of test instances] [<int> Band size ]";
    std::cout << "\n\tExample: ./svd_cuda benchmark 20 200 16 20" << std::endl;
    std::cout << "\n(2) Correctness Test: Compares test matrix and corresponding band and bidiagonal reductions" << std::endl;
    std::cout << "\t>> check [64|512|1024 Row/Column sizes]" << std::endl;
    std::cout << "\tExample: ./svd_cuda2 check 64\n" << std::endl;
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
                    std::string("/data/spencerrose/test_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
                    std::string(".bin");
            std::cout << "Reading file: " << filename << std::endl;
            A.read(filename);
            A.print();

            // Run CUDA band reduction
            std::cout << "\n\nCUDA-2 Test (Band):" << std::endl;
            csc586::gpu::cuda_brd_p1(A, band_size);
            A.print(16);

            // Compare with Baseline results
            std::cout << "\n\nBaseline Test (Band):" << std::endl;
            filename =
                    std::string("/data/spencerrose/band_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
                    std::string(".bin");
            band_check = csc586::gpu::Matrix<float>(size, size);
            band_check.read(filename);
            band_check.print(16);

            // Calculate Error
            auto error = A.mse(band_check, band_size);
            std::cout << "\n\nMSE of Band Reduction: " << error << std::endl;


            // Run CUDA bidiagonal reduction
            std::cout << "\n\nCUDA-2 Test (Bidiagonal):" << std::endl;
            csc586::gpu::brd(A);
            A.print(10);

            // Compare with Baseline results
            std::cout << "\n\nBaseline Test (Bidiagonal):" << std::endl;
            filename =
                    std::string("/data/spencerrose/bidiagonal_float_") + std::string(argv[2]) + std::string("_") + std::string(argv[2]) +
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

            // Step in size of matrix for each iteration
            auto step = size_t(atoi(argv[2]));
            // Number of steps
            auto n = size_t(atoi(argv[3]) + 1);
            // Number of test instances for benchmark
            auto n_test_instances = size_t(atoi(argv[4]));
            // Size of band (tile width)
            auto b_size = size_t(atoi(argv[5]));
            // Results array to write to file
            std::ostringstream vts;
            std::vector<int> x;
            std::vector<float> y;

            std::cout << "Benchmark: CUDA-2 Band Reduction" << std::endl;
            std::cout << "\tBand size: " << b_size << std::endl;
            std::cout << "\tStep size: " << step << std::endl;
            std::cout << "\tNumber of steps: " << n - 1 << std::endl;
            std::cout << "\tNumber of test instances: " << n_test_instances << std::endl;

            // Seed for the random number generator (current time)
            std::srand(static_cast< uint32_t >( std::time(0)));

            // Function references
            csc586::gpu::Matrix<T> (*brd_p1)(csc586::gpu::Matrix<T> &, const size_t) = csc586::gpu::cuda_brd_p1;

            // Run diagnostic loop for matrix size N = k * step
            std::cout << "Average time per CUDA-2 Band Reduction" << std::endl;
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

            }

            // Write benchmark results to file
            if (!x.empty() && !y.empty())
            {
                // Convert all but the last element to avoid a trailing ","
                std::copy(x.begin(), x.end()-1,
                          std::ostream_iterator<int>(vts, ", "));

                // Now add the last element with no delimiter
                vts << x.back();
                vts << "\n";

                std::copy(y.begin(), y.end()-1,
                          std::ostream_iterator<float>(vts, ", "));

                // Now add the last element with no delimiter
                vts << y.back();
            }
            auto filename = std::string("data/cuda_2_benchmark.csv");
            std::cout << "Writing results to file ... " << filename << std::endl;
            std::ofstream ftest;
            ftest.open(filename);
            ftest << vts.str();
            ftest.close();

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

