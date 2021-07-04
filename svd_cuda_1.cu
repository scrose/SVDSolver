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
#include "svd_cpu.h"  // GPU bidiagonal reduction
#include "timing.h"

namespace csc586 { // anonymous
    namespace gpu {

        /*
         * ===============================================
         * Constants
         * ===============================================
         */

        // Max number of threads per thread block = 2048.
        size_t const tile_size = 32u;
        // GPU block dimension (x, y, z)
        dim3 const dimBlock( tile_size, tile_size, 1u );
        // Minimum matrix size (width) for CUDA kernels
        size_t const min_width = 64u;


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
         * Output:
         *  - Result array (result) is overwritten with alpha*A + AB OR alpha*B + AB.
         * ===============================================
         */

        __global__
        void mm_kernel(float const *A, float const *B, float *result, size_t const m, size_t const n, size_t const p,
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
        void trans_kernel(float *A_T, float const *A, size_t const m, size_t const n )
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
         * Matrix Multiplication - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix product: C <- AB
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - Matrix <float> A (m x p matrix)
         *  - Matrix <float> B (p x n matrix)
         *  - Matrix <float> result (m x n matrix)
         *  - <float> alpha (constant in operation  A <- alpha*A + AB
         *  - <float> beta (constant in operation   B <- beta*B + AB
         * Output:
         *  - Result matrix is overwritten with alpha*A + AB OR alpha*B + AB.
         * ===============================================
         */

        void matmul( Matrix<float> &A, Matrix<float> &B, Matrix<float> &result,
                float const alpha = 0.0, float const beta = 0.0 , float const gamma = 1.0) {

            assert(A.ncols == B.nrows && "Matrix 1 col dim must match Matrix 2 row dim.");
            assert( !(alpha > 0 && beta > 0) && "Matrix self-addition must be for matrix A or B, not both.");

            // Parameters
            auto m = A.nrows;
            auto n = B.ncols;
            auto p = A.ncols;

            // Limit GPU matrix multiplication to cut-off size
            if (m * n <= min_width * min_width ) {
                result = A.mm(B);
                // A <- A + Result / B <- B + Result / Result <- gamma * Result
                if (alpha) A += result;
                if (beta) B += result;
                if (gamma) result *= gamma;
            }
            else {
                // calculate resultant matrix size
                auto const size_a = sizeof(float) * m * p;
                auto const size_b = sizeof(float) * p * n;
                auto const size_result = sizeof(float) * m * n;

                // Flatten matrices
                auto B_1d = B.flatten();
                auto A_1d = A.flatten();
                result = result.flatten();

                // Uninitialized pointers to memory on the GPU.
                float *dev_a, *dev_b, *dev_result;

                // Each of these allocations memory on the GPU for our input (first three) and output (last one).
                // Observe that we bind the allocated memory to the device pointers that we declared above.
                cudaMalloc((void **) &dev_a, size_a);
                cudaMalloc((void **) &dev_b, size_b);
                cudaMalloc((void **) &dev_result, size_result);

                // define number of blocks per grid
                dim3 dimGrid( 1, 1 );
                dimGrid.x = static_cast<int> (ceil(float( n + dimBlock.x - 1 )/float( dimBlock.x ) ) );
                dimGrid.y = static_cast<int> (ceil(float( m + dimBlock.y - 1 )/float( dimBlock.y ) ) );


                // Initiate a transfer of data between the host (CPU) and device (GPU).
                // Syntax: `(destination,source,size,direction)`.
                // `cudaMemcpyHostToDevice` constant denotes transferring data *to the GPU*.
                cudaMemcpy(dev_a, &A_1d[0][0], size_a, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_b, &B_1d[0][0], size_b, cudaMemcpyHostToDevice);

                // At last, we invoke the code on the GPU, using the data that we just transferred there.
                // It looks like normal C++ template code, except that the special syntax `<<<x,y>>>` configures
                // the assignment of threads to thread blocks.
                mm_kernel <<< dimGrid, dimBlock >>> (dev_a, dev_b, dev_result, m, n, p, alpha, beta, gamma);

                // Once the kernel has completed, we initiate a transfer of the result data *back to the CPU*.
                // Note that the `cudaMemcpyDeviceToHost` constant denotes transferring data *from the GPU*.
                cudaMemcpy(&result[0][0], dev_result, size_result, cudaMemcpyDeviceToHost);

                // Finally, because we are using old-fashioned mallocs, we need to manually clean-up after ourselves
                // These functions free memory that was allocated on the GPU/device.
                cudaFree(dev_a);
                cudaFree(dev_b);
                cudaFree(dev_result);

                // Restore rows and columns to result matrix
                result = result.reshape( m, n );

                // A <- A + Result / B <- B + Result
                if (alpha) A = result;
                if (beta) B = result;
            }

        }

    /*
         * ===============================================
         * Matrix Transpose - Host Wrapper
         * -----------------------------------------------
         * Evaluated matrix product: A' <- transpose(A)
         * CPU host function that contains built-in CUDA
         * functions to initialize the GPU kernel.
         *
         * Input:
         *  - Matrix <float> A (m x n matrix)
         *  - Matrix <float> A' (n x m matrix)
         * Output:
         *  - A' matrix is overwritten with transpose(A).
         * ===============================================
         */

        Matrix<float> transpose( Matrix<float> &A ) {

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Limit GPU matrix multiplication to cut-off size
            if (m * n <= min_width * min_width ) {
                auto A_T = A.transpose();
                return A_T;
            }
            else {
                // calculate resultant matrix size
                auto const size_a = sizeof(float) * m * n;

                // allocate transpose matrix
                auto A_T = Matrix<float>( 1, m * n );

                // Flatten matrices
                auto A_1d = A.flatten();

                // Uninitialized pointers to memory on the GPU.
                float *dev_a, *dev_a_t;

                // Each of these allocations memory on the GPU for our input (first three) and output (last one).
                // Observe that we bind the allocated memory to the device pointers that we declared above.
                cudaMalloc((void **) &dev_a, size_a);
                cudaMalloc((void **) &dev_a_t, size_a);

                // define number of blocks per grid
                dim3 dimGrid( 1, 1 );
                dimGrid.x = static_cast<int> (ceil(float( n + dimBlock.x - 1 )/float( dimBlock.x ) ) );
                dimGrid.y = static_cast<int> (ceil(float( m + dimBlock.y - 1 )/float( dimBlock.y ) ) );

                // Initiate a transfer of data between the host (CPU) and device (GPU).
                // Syntax: `(destination,source,size,direction)`.
                // `cudaMemcpyHostToDevice` constant denotes transferring data *to the GPU*.
                cudaMemcpy(dev_a, &A_1d[0][0], size_a, cudaMemcpyHostToDevice);

                // At last, we invoke the code on the GPU, using the data that we just transferred there.
                // It looks like normal C++ template code, except that the special syntax `<<<x,y>>>` configures
                // the assignment of threads to thread blocks.
                trans_kernel <<< dimGrid, dimBlock >>> ( dev_a_t, dev_a, m, n );

                // Once the kernel has completed, we initiate a transfer of the result data *back to the CPU*.
                // Note that the `cudaMemcpyDeviceToHost` constant denotes transferring data *from the GPU*.
                cudaMemcpy(&A_T[0][0], dev_a_t, size_a, cudaMemcpyDeviceToHost);

                // Finally, because we are using old-fashioned mallocs, we need to manually clean-up after ourselves
                // These functions free memory that was allocated on the GPU/device.
                cudaFree(dev_a);
                cudaFree(dev_a_t);

                // Restore rows and columns to transposed matrix and return
                A_T = A_T.reshape( n, m );
                return A_T;
            }
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
         * of a matrix A of size m x n for QR decomposition
         * Input:
         *  - A <Matrix> : m x n input matrix
         *  - S <Matrix> : m x m orthogonal matrix
         *  - V <Matrix> : m x m orthogonal matrix
         * ===============================================
         */

        void qr_apply_cuda( Matrix<float> &A, Matrix<float> &S, Matrix<float> &V ) {

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Limit GPU matrix multiplication to cut-off size
            if ( m * n <= min_width * min_width ) {

                qr_apply( A, S, V );

            }
            else {
                // calculate resultant matrix size
                auto const size_a = sizeof(float) * A.nrows * A.ncols;
                auto const size_s = sizeof(float) * S.nrows * S.ncols;
                auto const size_v = sizeof(float) * V.nrows * V.ncols;
                auto const size_v_t = sizeof(float) * size_v;
                auto const size_sv_t = sizeof(float) * S.nrows * V.nrows;
                auto const size_q = sizeof(float) * V.nrows * V.nrows;
                auto const size_q_t = size_q;

                // Flatten input matrices
                auto A_1d = A.flatten();
                auto S_1d = S.flatten();
                auto V_1d = V.flatten();

                // Uninitialized pointers to memory on the GPU.
                float *dev_a, *dev_s, *dev_v, *dev_v_t, *dev_sv_t, *dev_q, *dev_q_t, *dev_result;

                // define number of blocks per grid
                dim3 dimGrid( 1, 1 );
                dimGrid.x = static_cast<int> (ceil(float( n + dimBlock.x - 1 )/float( dimBlock.x ) ) );
                dimGrid.y = static_cast<int> (ceil(float( m + dimBlock.y - 1 )/float( dimBlock.y ) ) );

                // Invoke the code on the GPU, using the data that we just transferred there.
                // It looks like normal C++ template code, except that the special syntax `<<<x,y>>>` configures
                // the assignment of threads to thread blocks.

                // V' <- transpose(V)
                cudaMalloc((void **) &dev_v, size_v);
                cudaMalloc((void **) &dev_v_t, size_v_t);
                cudaMemcpy(dev_v, &V_1d[0][0], size_v, cudaMemcpyHostToDevice);
                trans_kernel <<< dimGrid, dimBlock >>> (dev_v_t, dev_v, V.nrows, V.ncols );

                // SV' <- SV'
                cudaMalloc((void **) &dev_s, size_s);
                cudaMalloc((void **) &dev_sv_t, size_sv_t);
                cudaMemcpy(dev_s, &S_1d[0][0], size_s, cudaMemcpyHostToDevice);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_s, dev_v_t, dev_sv_t, S.nrows, V.nrows, S.ncols, 0., 0., 1.);
                cudaFree(dev_v_t);
                cudaFree(dev_s);

                // Q <- VSV'
                cudaMalloc((void **) &dev_q, size_q);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_v, dev_sv_t, dev_q, V.nrows, V.nrows, V.ncols, 0., 0., 1.);
                cudaFree(dev_sv_t);
                cudaFree(dev_v);

                // Q' <- transpose(Q)
                cudaMalloc((void **) &dev_q_t, size_q_t);
                trans_kernel <<< dimGrid, dimBlock >>> (dev_q_t, dev_q, V.nrows, V.nrows );

                // R <- Q'A
                cudaMalloc((void **) &dev_a, size_a);
                cudaMalloc((void **) &dev_result, size_a);
                cudaMemcpy(dev_a, &A_1d[0][0], size_a, cudaMemcpyHostToDevice);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_q_t, dev_a, dev_result, V.nrows, A.ncols, V.nrows, 0., 1., 1.);
                cudaMemcpy(&A_1d[0][0], dev_result, size_a, cudaMemcpyDeviceToHost);
                cudaFree(dev_a);
                cudaFree(dev_q);
                cudaFree(dev_result);

                // Restore updated matrix to original dimensions
                A = A_1d.reshape(A.nrows, A.ncols);

            }
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
         * of a matrix A of size m x n for QR decomposition
         * Input:
         *  - A <Matrix> : m x n input matrix
         *  - S <Matrix> : m x m orthogonal matrix
         *  - V <Matrix> : m x m orthogonal matrix
         * ===============================================
         */

        void lq_apply_cuda( Matrix<float> &A, Matrix<float> &S, Matrix<float> &V) {

            // Parameters
            auto m = A.nrows;
            auto n = A.ncols;

            // Limit GPU matrix multiplication to cut-off size
            if ( m * n <= min_width * min_width ) {

                lq_apply( A, S, V );

            }
            else {
                // calculate resultant matrix size
                auto const size_a = sizeof(float) * A.nrows * A.ncols;
                auto const size_s = sizeof(float) * S.nrows * S.ncols;
                auto const size_v = sizeof(float) * V.nrows * V.ncols;
                auto const size_v_t = sizeof(float) * size_v;
                auto const size_sv = sizeof(float) * S.nrows * V.ncols;
                auto const size_p = sizeof(float) * A.nrows * A.ncols;

                // Flatten matrices
                auto A_1d = A.flatten();
                auto S_1d = S.flatten();
                auto V_1d = V.flatten();

                // Uninitialized pointers to memory on the GPU.
                float *dev_a, *dev_s, *dev_v, *dev_v_t, *dev_sv, *dev_p, *dev_result;

                // define number of blocks per grid
                dim3 dimGrid( 1, 1 );
                dimGrid.x = static_cast<int> (ceil(float( n + dimBlock.x - 1 )/float( dimBlock.x ) ) );
                dimGrid.y = static_cast<int> (ceil(float( m + dimBlock.y - 1 )/float( dimBlock.y ) ) );

                // V' <- transpose(V)
                cudaMalloc((void **) &dev_v, size_v);
                cudaMalloc((void **) &dev_v_t, size_v_t);
                cudaMemcpy(dev_v, &V_1d[0][0], size_v, cudaMemcpyHostToDevice);
                trans_kernel <<< dimGrid, dimBlock >>> (dev_v_t, dev_v, V.nrows, V.ncols );

                // SV <- SV
                cudaMalloc((void **) &dev_s, size_s);
                cudaMalloc((void **) &dev_sv, size_sv);
                cudaMemcpy(dev_s, &S_1d[0][0], size_s, cudaMemcpyHostToDevice);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_s, dev_v, dev_sv, S.nrows, V.ncols, S.ncols, 0., 0., 1.);
                cudaFree(dev_v);
                cudaFree(dev_s);

                // P <- V'SV
                cudaMalloc((void **) &dev_p, size_p);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_v_t, dev_sv, dev_p, A.nrows, A.ncols, V.nrows, 0., 0., 1.);
                cudaFree(dev_sv);
                cudaFree(dev_v_t);

                // Result <- AP
                cudaMalloc((void **) &dev_a, size_a);
                cudaMalloc((void **) &dev_result, size_p);
                cudaMemcpy(dev_a, &A_1d[0][0], size_a, cudaMemcpyHostToDevice);
                mm_kernel <<< dimGrid, dimBlock >>> (dev_a, dev_p, dev_result, A.nrows, A.ncols, A.ncols, 1., 0., 1.);

                // Once the kernel has completed, we initiate a transfer of the result data *back to the CPU*.
                // Note that the `cudaMemcpyDeviceToHost` constant denotes transferring data *from the GPU*.
                cudaMemcpy(&A_1d[0][0], dev_result, size_a, cudaMemcpyDeviceToHost);

                // Finally, because we are using old-fashioned mallocs, we need to manually clean-up after ourselves
                // These functions free memory that was allocated on the GPU/device.
                cudaFree(dev_a);
                cudaFree(dev_p);
                cudaFree(dev_result);

                // Restore rows and columns to result matrix
                A = A_1d.reshape( m, n );

            }
        }



        /*
        * ===============================================
        * Householder Reflection
        * -----------------------------------------------
        * Input: Column vector w
        * Output: x (projection vector); tau (scaling factor)
        * ===============================================
        */
        Reflection<float> householder( Matrix<float> w ) {

            assert( w.ncols == 1 && "Householder applied to column vectors only." );

            auto x = w.col_slice(0, 0, w.nrows);
            auto s = -std::copysign(1, x[0]);
            // calculate Euclidean normalization of vector
            auto norm_x = norm(x);

            // w = ( x − s||x||e1 ) / u1
            auto u1 = x[0] - s * norm_x;
            w *= 1./u1;
            w[0][0] = 1.;
            float tau = -s * u1 / norm_x;

            // compute w'
            auto w_T = transpose(w);

            Reflection<float> result = { w, w_T, tau };
            return result;

        }



        /*
          * ===============================================
          * Compact YT representation of Householder reflectors
          * Hk...H3.H2.H1 = I - VSV'
          * -----------------------------------------------
          * See Schreiber and Van Loan (1989)
          *
          * Input:
          *  - tau <float>: Householder scalar factor
          *  - S <Matrix>: k x k Triangular matrix T
          *  - V <Matrix>: n x k Householder vectors Y
          * Output:
          *
          *  - S_k+1 = [ S_k    −tau_k+1.S_k.V_k'.v_k+1 ]
          *            [  0              -tau_k+1       ]
          * ===============================================
          */

        void hholder_compact( const size_t j, const float& tau, Matrix <float>& S, Matrix <float>& V ) {
            if ( j == 0u ) {
                S[0][0] = -tau;
            }
            else {
                auto S_k = S.slice(Slice{0, j, 0, j});
                auto v = V.slice(Slice{0, V.nrows, j, j+1});
                auto V_k = V.slice(Slice{0, V.nrows, 0, j}).transpose();
                auto z = Matrix<float>( V_k.nrows, v.ncols);
                matmul( V_k, v, z);
                matmul( S_k, z, z, 0, 0, -tau);
                // copy T_01 = z
                S.copy(z, Slice{0, z.nrows, S_k.ncols, S_k.ncols + z.ncols});

                // copy T_11 = -tau
                S[S_k.nrows][S_k.ncols] = -tau;
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
        * Input:
        *  - A <Matrix> : m x n input matrix
        *  - S <Matrix> : m x m upper triangular matrix
        *  - V <Matrix> : m x n lower triangular matrix
        * ===============================================
        */


        void qr_cuda( Matrix <float>& A, Matrix <float>& S, Matrix <float>& V ) {

            // Initialize block compact householder matrices
            auto m = A.nrows;
            auto n = A.ncols;
            auto Y = Matrix<float>(n, n);
            auto y = Matrix<float>(1, n); // 1 x n
            auto result = Matrix<float>( m, n ); // m x n

            // Reduction of rectangular matrix to upper triangular form
            for (auto j = 0u; j < std::min(n, m); ++j) {

                // Reduce column j of A
                auto R = A;
                // R -= V.mm(Y);
                matmul( V, Y, result, 0, 0);
                R -= result;

                // compute Householder H_j from column A_j
                auto A_j = R.slice(j, m, j, j + 1);
                auto H_j = householder(A_j);

                // Compute Y(i+1:n,i): y_i = tau * R'v
                auto A_trail = R.slice(j, m, j, n).transpose();
                matmul( A_trail, H_j.w, y, 0, 0, H_j.tau);

                // Update matrices V, Y
                V.copy(H_j.w, Slice{j, m, j, j + 1});
                Y.copy(y.transpose(), Slice{j, j + 1, j, n});

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(j), H_j.tau, S, V );

            }

            // update matrix with A <- A - VY
            matmul( V, Y, result, 0, 0);
            A -= result;
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
        * compact WY technique
        * Input:
        *  - A <Matrix> : m x n input matrix
        *  - S <Matrix> : m x m upper triangular matrix
        *  - V <Matrix> : m x n upper triangular matrix
        * ===============================================
        */


        void lq_cuda( Matrix <float>& A, Matrix <float>& S, Matrix <float>& U ) {

            // Initialize block compact householder matrices
            auto m = A.nrows;
            auto n = A.ncols;
            auto X = Matrix<float>(m, m);
            auto x = Matrix<float>(m, 1); // m x 1
            auto U_T = Matrix<float>(n, m);
            auto result = Matrix<float>( m, n ); // m x n

            // Reduction of rectangular matrix to lower triangular form
            for (auto i = 0u; i < std::min(n, m); ++i) {

                // Reduce row j of A = A - XU'
                auto L = A;
                matmul( X, U, result, 0, 0);
                L -= result;

                // compute Householder H_i to eliminate right of diagonal
                auto A_i = L.slice(i, i + 1, i, n);
                auto H_i = householder(A_i.transpose());

                // x = tau * L'u
                auto A_trail = L.slice(i, m, i, n);
                matmul( A_trail, H_i.w, x, 0, 0, H_i.tau);

                // Update matrices U, X
                X.copy(x, Slice{i, m, i, i + 1});
                U.copy(H_i.w.transpose(), Slice{i, i + 1, i, n});
                U_T = U.transpose();

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(i), H_i.tau, S, U_T);

            }

            // Update matrix: A <- A - XU'
            matmul( X, U, result, 0, 0);
            A -= result;

        }



        /*
        * ===============================================
        * CUDA Blocked Band Reduction <cuda_brd_p1>
        * -----------------------------------------------
         * Dense matrix -> Banded matrix (Stage I of two-stage process)
         * -----------------------------------------------
         * Computes banded bidiagonal matrix B = U1'*A*V1 using
         * QR and LQ transformations to upper and lower diagonals
         * Input: Matrix <float> A (m x n matrix)
         * Output:
         *  - Matrix <float> B (banded bidiagonal m x n matrix)
         *  - Matrix <float> U1 (left-side orthogonal matrix)
         *  - Matrix <float> V1 (right-side orthogonal matrix)
         * ===============================================
         */

        Matrix<float> cuda_brd_p1( Matrix <float>& A, size_t const b_size ) {

            // Matrix dimensions
            auto m = A.nrows;
            auto n = A.ncols;

            // Initialize container matrices
            auto S = Matrix<float>(b_size, b_size);
            auto A_trail = A;

            // Iterate over blocks of A (size: b_size)
            for ( auto k = 0u; k < n; k += b_size ) {

                // Extract QR panel
                auto A_panel_qr = A_trail.slice( 0, A_trail.nrows, 0, b_size );

                // Extract trailing matrix
                A_trail = A_trail.slice( 0, A_trail.nrows, b_size, A_trail.ncols );

                // Initialize compact householder matrices
                auto V = Matrix<float>(A_panel_qr.nrows, b_size);

                // (Step 1) QR Reduction of left panel
                qr_cuda( A_panel_qr, S, V );

                // Update trailing matrix QA = (I − V T V')' A
                qr_apply_cuda( A_trail, S, V );

                // Copy QR reduced panel to A
                A.copy( A_panel_qr, Slice{k, m, k, k + b_size} );

                if ( k + b_size < n - 1 ) {

                    // Extract QR panel
                    auto A_panel_lq = A_trail.slice( 0, b_size, 0, A_trail.ncols );
                    // Extract trailing matrix
                    A_trail = A_trail.slice( b_size, A_trail.nrows, 0, A_trail.ncols );

                    // Initialize compact householder matrices
                    auto U = Matrix<float>(b_size, A_panel_lq.ncols);

                    // (Step 2) LQ Reduction of right panel
                    lq_cuda( A_panel_lq, S, U );

                    // Update trailing matrix AQ' =  A (I − V T V')'
                    lq_apply_cuda( A_trail, S, U );

                    // Copy LQ reduced panel to A
                    A.copy( A_panel_lq, Slice{ k, k + b_size, k + b_size, n} );

                }
                // A <- residual trailing matrix
                A.copy( A_trail, Slice{ m - A_trail.nrows, m, n - A_trail.ncols, n } );
            }

            return A;
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
    std::cout << "Options for CUDA-1 Testing" << std::endl;
    std::cout << "\n(1) Run benchmark tests for CUDA-1 band reduction." << std::endl;
    std::cout << "\t>> benchmark [<int> Step size] [<int> Number of steps] [<int> Number of test instances] [<int> Band size ]";
    std::cout << "\n\tExample: ./svd_cuda benchmark 20 200 16 20" << std::endl;
    std::cout << "\n(2) Correctness Test: Compares test matrix and corresponding band and bidiagonal reductions" << std::endl;
    std::cout << "\t>> check [64|512|1024 Row/Column sizes]" << std::endl;
    std::cout << "\tExample: ./svd_cuda1 check 64\n" << std::endl;
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
            std::cout << "\n\nCUDA-1 Test (Band):" << std::endl;
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
            std::cout << "\n\nCUDA-1 Test (Bidiagonal):" << std::endl;
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

            std::cout << "Benchmark: CUDA-1 Band Reduction" << std::endl;
            std::cout << "\tBand size: " << b_size << std::endl;
            std::cout << "\tStep size: " << step << std::endl;
            std::cout << "\tNumber of steps: " << n - 1 << std::endl;
            std::cout << "\tNumber of test instances: " << n_test_instances << std::endl;

            // Seed for the random number generator (current time)
            std::srand(static_cast< uint32_t >( std::time(0)));

            // Function references
            csc586::gpu::Matrix<T> (*brd_p1)(csc586::gpu::Matrix<T> &, const size_t) = csc586::gpu::cuda_brd_p1;

            // Run diagnostic loop for matrix size N = k * step
            std::cout << "Average time per CUDA-1 Band Reduction" << std::endl;
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
            auto filename = std::string("data/cuda_1_benchmark.csv");
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