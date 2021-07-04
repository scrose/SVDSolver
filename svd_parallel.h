

#ifndef CS586_PSVD
#define CS586_PSVD

/**
* **********************************************
* Parallel Singular Value Decomposition
* **********************************************
 * CSC 586B - Spring 2020 - Interim Project #2
 * Author: Spencer Rose
 * *********************************************
 * Applies two-step SVD reduction of mxn matrix A
 * tto the form A = U\SigmaV^T where the columns
 * of U form an nxn orthonormal matrix; the rows
 * of V^T form an nxn orthonormal matrix, and \Sigma
 * is an m×n diagonal matrix with positive real
 * entries known as the singular values of A.
 *
* Data Structures:
*  - Bidiagonal{}: matrix diagonal and bidiagonal
*  - Reflection{}: Householder reflector
 *
* Input: Matrix (n x m)
* Output: Bidiagonal matrix as diagonal and super-diagonal vectors (n)
 *
* Functions:
*  - brd_p1(): Dense to Band matrix reduction
*  - brd_p2(): Band to bidiagonal reduction
* **********************************************
**/

#include <iostream>
#include <iomanip>
#include <cassert>
#include <tuple>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <random>
#include <functional>
#include <omp.h>
#include "matrix.h"  // matrix class with operators
#include "svd_serial.h"  // SVD methods with operators

namespace csc586 {
    namespace parallel {

        // Print slice values
        void pt( Slice t ) {
            std::cout << "\n\n ==== t: [" << t.i1 << ':' << t.i2 << ',' << t.j1 << ':' << t.j2 << ']' << std::endl << std::endl;
        }


        // Data type for simple thread tracker / scheduler
        struct Tracker {
            std::vector<bool> iterations; // tile boolean array
            std::vector<std::vector<bool>> tasks;
            const size_t steps = 4u; // steps for each decomposition

            // initialize tracker for matrix parameters
            void init( const size_t nbt ) {
                iterations.resize( nbt, false );
                tasks.resize( steps, iterations);
            }
            // Update kth tile task at step
            void update( const size_t& step, const size_t& k )
            {
                tasks[k][step] = true;
            }
            // Update kth tile task at step
            bool is_complete( const size_t& step, const size_t& k )
            {
                return tasks[k][step];
            }
        };

        /*
          * ===============================================
          * Compact YT representation of Householder reflectors
          * Hk...H3.H2.H1 = I - VSV'
          * -----------------------------------------------
          * See Schreiber and Van Loan (1989)
          *
          * Input:
          *  - tau <T>: Householder scalar factor
          *  - S <Matrix>: k x k Triangular matrix T
          *  - V <Matrix>: n x k Householder vectors Y
          * Output:
          *
          *  - S_k+1 = [ S_k    −tau_k+1.S_k.V_k'.v_k+1 ]
          *            [  0              -tau_k+1       ]
          * ===============================================
          */

        template<typename T>
        void hholder_compact( const size_t j, const T& tau, Matrix <T>& S, Matrix <T>& V ) {
            if ( j == 0u ) {
                S[0][0] = -tau;
            }
            else {
                auto S_k = S.slice(Slice{0, j, 0, j});
                auto v = V.slice(Slice{0, V.nrows, j, j+1});
                auto V_k = V.slice(Slice{0, V.nrows, 0, j}).transpose(true);
                auto z = V_k.mm(v, true);
                z = S_k.mm(z, true);
                z *= -tau;
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
        *  - V <Matrix> : m x n upper triangular matrix
        * ===============================================
        */

        template<typename T>
        void qr( Matrix <T>& A, Matrix <T>& S, Matrix <T>& V ) {

            // Initialize block compact householder matrices
            auto m = A.nrows;
            auto n = A.ncols;
            auto Y = Matrix<T>(n, n);
            auto y = Matrix<T>(1, n); // 1 x n

            // Reduction of rectangular matrix to upper triangular form
            for (auto j = 0u; j < std::min(n, m); ++j) {

                // Reduce column j of A
                auto R = A;
                R -= V.mm(Y, true);

                // compute Householder H_j from column A_j
                auto A_j = R.slice(j, A.nrows, j, j + 1);
                auto H_j = serial::householder(A_j);

                // Compute Y(i+1:n,i): y_i = tau * R'v
                auto A_trail = R.slice(j, A.nrows, j, A.ncols).transpose(true);
                y = A_trail.mm(H_j.w, true);
                y *= H_j.tau;

                // Update matrices V, Y
                V.copy(H_j.w, Slice{j, m, j, j + 1});
                Y.copy(y.transpose(true), Slice{j, j + 1, j, n});

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(j), H_j.tau, S, V);

            }

            // update matrix with A <- A - VY'
            A -= V.mm(Y, true);

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

        template<typename T>
        void lq( Matrix <T>& A, Matrix <T>& S, Matrix <T>& U ) {

            // Initialize block compact householder matrices
            auto m = A.nrows;
            auto n = A.ncols;
            auto X = Matrix<T>(m, m);
            auto x = Matrix<T>(m, 1); // m x 1
            auto U_T = Matrix<T>(n, m);

            // Reduction of rectangular matrix to lower triangular form
            for (auto i = 0u; i < std::min(n, m); ++i) {

                // Reduce row j of A = A - XU'
                auto L = A;
                L -= X.mm(U, true);

                // compute Householder H_i to eliminate right of diagonal
                auto A_i = L.slice(i, i + 1, i, n);
                auto H_i = serial::householder(A_i.transpose(true));

                // x = tau * L'u
                auto A_trail = L.slice(i, m, i, n);
                x = A_trail.mm(H_i.w, true);
                x *= H_i.tau;

                // Update matrices U, X
                X.copy(x, Slice{i, m, i, i + 1});
                U.copy(H_i.w.transpose(true), Slice{i, i + 1, i, n});
                U_T = U.transpose(true);

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(i), H_i.tau, S, U_T);

            }
            // update matrix with A <- A - XU'
            A -= X.mm(U, true);

        }


    /*
        * ===============================================
        * Kernel: Apply Householder for QR Decomposition
        * -----------------------------------------------
        * Subroutine applies S and V to matrix A
        * of a matrix A of size m x n for QR decomposition
        * Input:
        *  - A <Matrix> : m x n input matrix
        *  - S <Matrix> : m x m orthogonal matrix
        *  - V <Matrix> : m x m orthogonal matrix
        * ===============================================
        */

        template<typename T>
        void qr_apply( Matrix <T>& A, Matrix <T>& S, Matrix <T>& V ) {

            // Compute Q = I + VSV' from compact WY format
            Matrix<T> Q;
            auto V_T = V.transpose(true);
            Q = S.mm( V_T , true);
            Q = V.mm( Q , true);
            Q = Q.transpose(true);

            // A <- A + (VSV')^T.A
            A += Q.mm(A, true);
        }


        /*
        * ===============================================
        * Kernel: Apply Householder for LQ Decomposition
        * -----------------------------------------------
        * Subroutine applies S and V to matrix A
        * of a matrix A of size m x n for QR decomposition
        * Input:
        *  - A <Matrix> : m x n input matrix
        *  - S <Matrix> : m x m orthogonal matrix
        *  - V <Matrix> : m x m orthogonal matrix
        * ===============================================
        */

        template<typename T>
        void lq_apply( Matrix <T>& A, Matrix <T>& S, Matrix <T>& V ) {

            // Compute Q = I + VSV' from compacted householders
            Matrix<T> P;
            auto V_T = V.transpose(true);
            P = S.mm(V, true);
            P = V_T.mm(P, true);

            // A <- A + A.(VSV')
            A += A.mm(P, true);
        }


        /*
         * ===================================================
        * Kernels: Dense -> Band Reduction Kernels
        * ===================================================
        * */

        /* Kernel: Tile Factorization (1 and 2 tiles)
         * ---------------------------------------------------
         * QR Factorization of single tile
         *
         * */
        template<typename T>
        void factor_1tile(
                void (*transform)(Matrix <T>&, Matrix <T>&, Matrix <T>&),
                Matrix <T>& A, const size_t i, const size_t j, const size_t nbt, Matrix <T>& R, Matrix <T>& S, Matrix <T>& V ) {

            // Extract tile A_ij
            R = A.get_tile(i, j, nbt);

            // R_ij <- (I + VSV')A_ij
            transform(R, S, V);

            // Update A_ij <- R_ij
            A.set_tile(R, i, j, nbt);
        }

        template<typename T>
        void factor_2tile(
                void (*transform)(Matrix <T>&, Matrix <T>&, Matrix <T>&),
                Matrix <T>& A, const size_t i1, const size_t j1, const size_t i2, const size_t j2, const size_t nbt, const size_t t_size,
                Matrix <T>& R, Matrix <T>& S, Matrix <T>& V, const bool rowwise = true ) {

            // concatenate R_kj with A_ij
            auto A_i2j2 = A.get_tile(i2, j2, nbt);

            // concatenate tile 1 with tile 2
            if ( rowwise ) {
                R.row_concat(A_i2j2);
                transform(R, S, V);
                A_i2j2 = R.slice(t_size, t_size + t_size, 0, t_size);

            }
            else {
                R.col_concat(A_i2j2);
                transform(R, S, V);
                A_i2j2 = R.slice(0, t_size, t_size, t_size + t_size);
            }

            // Extract R_i2j2
            R = R.slice(0, t_size, 0, t_size);

            // Update A_i1j1 <- R_i1j1,
            A.set_tile(R, i1, j1, nbt);
            // Update A_i2j2 <- R_i2j2
            A.set_tile(A_i2j2, i2, j2, nbt);
        }

        /* Kernels : Tile Application Reduction (1 and 2 tiles)
         * ---------------------------------------------------
         * Apply Q orthogonal factorization to single tile
         *
         * */
        template<typename T>
        void apply_1tile(
                void (*apply_q)(Matrix <T>&, Matrix <T>&, Matrix <T>&),
                Matrix <T>& A, const size_t i, const size_t j, const size_t nbt, Matrix <T>& S, Matrix <T>& V ) {

            // A_ij <- H.A_ij or A_ij.G
            auto A_ij = A.get_tile(i, j, nbt);

            // Apply QR transformation to row k tiles
            apply_q(A_ij, S, V);

            // Update A_ij <- Q(A_kj)
            A.set_tile(A_ij, i, j, nbt);
        }


        template<typename T>
        void apply_2tile(
                void (*transform)(Matrix <T>&, Matrix <T>&, Matrix <T>&),
                Matrix <T>& A, const size_t i1, const size_t j1, const size_t i2, const size_t j2, const size_t nbt,
                const size_t t_size, Matrix <T>& S, Matrix <T>& V, const bool rowwise = true )  {

            // concatenate R_kj with A_ij
            auto TS = A.get_tile(i1, j1, nbt);
            auto Ai2j2 = A.get_tile(i2, j2, nbt);

            // concatenate tile 1 with tile 2
            if ( rowwise ) {
                TS.row_concat(Ai2j2);
                transform(TS, S, V);
                Ai2j2 = TS.slice(t_size, t_size + t_size, 0, t_size);
            }
            else {
                TS.col_concat(Ai2j2);
                transform(TS, S, V);
                Ai2j2 = TS.slice(0, t_size, t_size, t_size + t_size);
            }

            // Update A_ij <- A_ij
            A.set_tile(Ai2j2, i2, j2, nbt);

            // Update A_i1j1 <- R_i1j1
            auto A_i1j1 = TS.slice(0, t_size, 0, t_size);
            A.set_tile(A_i1j1, i1, j1, nbt);

        }



        /*
         * ===============================================
         * Banded Bidiagonal Reduction (Großer, et al., 1999)
         * Dense matrix -> Banded matrix (Stage I of two-stage process)
         * -----------------------------------------------
         * Computes banded bidiagonal matrix B = U1'*A*V1 using
         * QR and LQ transformations to upper and lower diagonals
         * Input: Matrix <T> A (m x n matrix)
         * Output:
         *  - Matrix <T> B (banded bidiagonal m x n matrix)
         *  - Matrix <T> U1 (left-side orthogonal matrix)
         *  - Matrix <T> V1 (right-side orthogonal matrix)
         * ===============================================
         */

        template<typename T>
        Matrix<T> brd_p1( Matrix <T>& A, size_t const t_size ) {

            const auto m = A.nrows;
            const size_t& nbt = size_t (m / t_size);

            // Initialize S_kk and V_kk
            auto S_kk = Matrix<T>(t_size, t_size);
            auto V_kk = Matrix<T>(t_size, t_size);

            // Initialize S_ik and V_ik
            auto S_ik = Matrix<T>(t_size, t_size);
            auto V_ik = Matrix<T>(2 * t_size, t_size);

            // Initialize S_ik and V_ik
            auto S_ki = Matrix<T>(t_size, t_size);
            auto V_ki = Matrix<T>(t_size, 2 * t_size);

            // Initialize S_kk+1 and V_kk+1
            auto S_kk1 = Matrix<T>(t_size, t_size);
            auto V_kk1 = Matrix<T>(t_size, t_size);

            auto R_kk = Matrix<T>(t_size, t_size);
            auto R_kk1 = Matrix<T>(t_size, t_size);


            // Reduce across tile columns
            for (auto k = 0u; k < nbt; ++k) {


                // ====== [QR - Step 1] =======

                // Factor the top left OR bottom right diagonal tile
                if (k == 0u || k == nbt - 1 ) {
                    factor_1tile(qr, A, k, k, nbt, R_kk, S_kk, V_kk);
                }


                // ====== [QR - Step 2] =======

                // Apply V_kk, S_kk along row k
                #pragma omp for schedule( static )
                for (auto j = k + 1; j < nbt; ++j) {

                    apply_1tile(qr_apply, A, k, j, nbt, S_kk, V_kk);

                    // factor initial TS block to accelerate Step 3
                    // --------------------------------------------
                    if ( j == k + 1 ) {
                        factor_2tile( qr, A, k, k, j, k, nbt, t_size, R_kk, S_ik, V_ik, true );
                    }
                }

                // ====== [QR - Step 3] =======

                // elimination along column k
                for (auto i = k + 1; i < nbt; ++i) {

                    // Start at column k + 2 (Note: first block created in QR Step 2)
                    if ( i > k + 1 ) {
                        // concatenate R_kk with A_ik
                        factor_2tile( qr, A, k, k, i, k, nbt, t_size, R_kk, S_ik, V_ik, true );
                    }


                    // ====== [QR - Step 4] =======
                    // apply update to trailing matrix
                    #pragma omp parallel for schedule( static)
                        for (auto j = k + 1; j < nbt; ++j) {
                        apply_2tile(qr_apply, A, k, j, i, j, nbt, t_size, S_ik, V_ik, true);

                            // factor next LQ diagonal block to accelerate LQ Step 2
                            if ( j == k + 1 && i == nbt - 1 && k < nbt - 1 ) {
                                factor_1tile(lq, A, k, k + 1, nbt, R_kk1, S_kk1, V_kk1);
                            };
                        }
                }


                    // ====== Phase 2: LQ Decomposition =======
                    if (k < nbt - 1) {

                        // ====== [LQ - Step 1] =======
                        // Apply LQ reduction to tile k,k+1 (completed in step 4)


                        // ====== [LQ - Step 2] =======
                        // Apply LQ reduction along column k + 1
                        #pragma omp parallel for schedule( static )
                        for (auto j = k + 1; j < nbt; ++j) {

                            apply_1tile(lq_apply, A, j, k + 1, nbt, S_kk1, V_kk1);

                            // factor initial TS block to accelerate Step 3
                            // --------------------------------------------
                            if ( j == k + 1 && k + 2 < nbt ) {
                                factor_2tile(lq, A, k, k + 1, k, k + 2, nbt, t_size, R_kk1, S_ki, V_ki, false);
                            }
                        }


                        // ====== [LQ - Step3] =======
                        for (auto i = k + 2; i < nbt; ++i) {

                            // Start at k + 3 (Note: first block created in QR Step 2)
                            if ( i > k + 2 ) {
                                factor_2tile(lq, A, k, k + 1, k, i, nbt, t_size, R_kk1, S_ki, V_ki, false);
                            }

                            // ====== [LQ - Step 4] =======
                            #pragma omp parallel for schedule( static )
                            for (auto j = k + 1; j < nbt; ++j) {
                                apply_2tile(lq_apply, A, j, k + 1, j, i, nbt, t_size, S_ki, V_ki, false );

                                if ( j == k + 1 && i == nbt - 1 ) {
                                    factor_1tile(qr, A, j, j, nbt, R_kk, S_kk, V_kk);
                                }
                            }
                        }
                    }

            }
            return A;
        }


        /*
          * ===============================================
          * Band -> Bidiagonal Reduction
          * Stage II of two-stage bidiagonal reduction
          * -----------------------------------------------
          * See Haidar et al. (2012).
          *
          * Input:
          *  - B <Matrix> : m x n Banded Matrix
          *  - nbt <size_t> : band size
          * Output:
          *  - B
          * ===============================================
        */


        /*
         * ===================================================
         * Band -> Diagonal Reduction Kernels
         * ===================================================
         * */

        /* Kernel : Band Reduction Top
         * ---------------------------------------------------
         * Triggers start of each band reduction sweep by successive
         * element-wise eliminations of the extra non-zero entries
         * within a single column. It then applies all the left
         * updates creating single bulges, immediately zeroed and then
         * followed by the right updates on the corresponding data
         * block loaded into the cache memory.
         *
         * */
        template<typename T>
        void band_rd_top( Matrix <T>& A, const size_t i, const size_t b_size, Slice& t ) {

            // Reduce row i above diagonal
            // {w, tau} = householder( A’[i+1:i+shift-1,i+1:i+shift] )
            auto A_t = A.slice( t );
            auto x = A_t.slice(0, 1, 0, A_t.ncols);
            auto H = serial::householder(x.transpose(true)).transform;
            A_t = A_t.mm( H, true );
            A.copy( A_t, t );

            // Reduce col i + 1 below diagonal
            // {w, tau} = householder( A[i:i+shift,i+1:i+2] )
            auto end_j  = std::min( i + b_size + b_size - 1, A.ncols );
            t = Slice{ t.i1 + 1, t.i2, i + 1, end_j };
            A_t = A.slice( t );
            x = A_t.slice(0, A_t.nrows, 0, 1);
            H = serial::householder(x).transform;
            A_t = H.mm(A_t, true);
            A.copy( A_t, t );

        }


        /* Kernel : Band Reduction Right
         * ----------------------------------------------------
         * Applies all right updates from previous kernel,
         * either band_rd_top or band_rd_left.
         * This generates single bulges immediately eliminated
         * by left transformations.
         */
        template<typename T>
        void band_rd_right( Matrix <T>& A, Slice& t ) {

            // Eliminate top row entries of tile
            auto A_t = A.slice( t );
            auto x = A_t.slice(0, 1, 0, A_t.ncols);
            auto H = serial::householder(x.transpose(true)).transform;
            A_t = A_t.mm(H, true);
            A.copy( A_t, t );
        }

        /* Kernel : Band Reduction Left
         * ----------------------------------------------------
         * Applies all left updates from previous kernel.
         * Creates single bulge out of the diagonal that is
         * eliminated; applies corresponding right updates.
         */
        template<typename T>
        void band_rd_left( Matrix <T>& A, Slice& t ) {
            // Eliminate leftmost column entries
            auto A_t = A.slice( t );
            auto x = A_t.slice(0, A_t.nrows, 0, 1);
            auto H = serial::householder(x).transform;
            A_t = H.mm(A_t, true);
            A.copy( A_t, t );
        }


        /*
         * * ===================================================
         * Main Band-to-Bidiagonal Reduction Loop
         * =====================================================
         * Main Band-to-Bidiagonal Reduction Loop (Haider, et al., 2013).
         * Completes Banded matrix -> Bidiagonal matrix (Stage II of
         * two-stage process). Computes bidiagonal matrix B = U1'.A.V1
         * using Householder transformations to upper and lower diagonals
         * of banded matrix.
         * =====================================================
        * */

        template<typename T>
        serial::Bidiagonal<T> brd_p2( Matrix <T>& A, size_t b_size = 0u ) {
            auto m = A.nrows;
            auto n = A.ncols;
            Slice t_left = {0,0,0,0};
            Slice t_right = {0,0,0,0};
            size_t end_i, end_j, start_j, end_j3;
            //auto threshold = 1e-6;

            b_size += 1;

            // iterate over columns of A

            for (auto i = 0u; i < n - 1; ++i) {

                // ======== Task 1 =========
                // compute tile dimensions for row elimination
                end_i = std::min( i + b_size, A.nrows );
                end_j  = std::min( i + b_size, A.ncols );
                t_left = {i, end_i, i + 1, end_j};

                // Reduce top diagonal (returns householder reflector)
                band_rd_top(A, i, b_size, t_left);

                // calculate number of Tasks 2 and 3 iterations
                auto nbtx = size_t ( std::ceil( ( n - t_left.j2 ) / ( b_size - 1 ) ) );

                // Iterate over remaining tiles

                for (auto k = 0u; k < nbtx + 1; ++k) {

                    // Initialize Task 2 and 3 target slices
                    end_i = std::min(t_left.i2 + b_size - 1, m);
                    start_j = std::min(t_left.j1 + b_size - 1, n);
                    end_j3 = std::min(t_left.j2 + b_size - 1, n);

                    t_right = {t_left.i1, end_i, start_j, t_left.j2};
                    t_left = {t_left.i2, end_i, start_j, end_j3};


                    // ======== Task 2 =========
                    if (t_right.j2 > t_right.j1)
                        band_rd_right(A, t_right);


                    // ======== Task 3 =========
                    if (t_left.j2 > t_left.j1)
                        band_rd_left(A, t_left);

                }
            }

            serial::Bidiagonal<T> B = { A.diag(), A.diag(1) };
            return B;


        }




    } // namespace parallel
} // namespace csc586
#endif // CS586_PSVD
