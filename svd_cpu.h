

#ifndef CS586_GPUSVD
#define CS586_GPUSVD

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
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <functional>
#include "matrix_gpu.h"  // matrix class with operators

namespace csc586 {
    namespace gpu {


        /*
        * ===============================================
        * Parameters for bidiagonal matrix B
        * -----------------------------------------------
        * Members:
        *  - d: diagonal of B
        *  - e: superdiagonal of B
        * Methods:
         *  - slice()
         *  - print()
        * ===============================================
        */
        template <typename T>
        struct Bidiagonal {
            std::vector<T> d; // diagonal
            std::vector<T> e; // superdiagonal

            // Returns slice of bidiagonal matrix
            Bidiagonal<T> slice( const size_t d_start, const size_t d_end, const size_t e_start, const size_t e_end ) {
                Bidiagonal<T> tmp;
                tmp.d.resize(d_end - d_start + 1);
                tmp.e.resize(e_end - e_start + 1);
                std::copy(d.begin() + d_start, d.end() - (d.size() - d_end) + 1, tmp.d.begin());
                std::copy(e.begin() + e_start, e.end() - (e.size() - e_end) + 1, tmp.e.begin());
                return tmp;
            }
            // Prints bidiagonal to console
            void print( const size_t truc = 10, const uint32_t precision = 4u )
            {
                std::cout << std::fixed;
                std::cout << std::setprecision(precision);
                std::cout << "\n-------\nBidiagonal capacity: " << 2*e.capacity() << std::endl;
                std::cout << "Elements: " << 2*d.size() << "]" << std::endl;
                std::cout << "Size[Bytes]: " << 2*sizeof(d) << 'b' << std::endl;
                // Print diagonal
                std::cout << "\nDiagonal:\n" << std::endl;
                for( auto i = 0u; i <= truc && i < d.size(); ++i )
                { // iterate rows
                    if ( i == truc ) { // add ellipsis for truncated rows
                        std::cout << " ... " << std::endl;
                        i = d.size() - 1u;
                    }
                    std::cout << ' ' << d[i] << ' ';

                }
                // Print superdiagonal
                std::cout << "\nSuperdiagonal:\n" << std::endl;
                for( auto i = 0u; i <= truc && i < e.size(); ++i )
                { // iterate rows
                    if ( i == truc ) { // add ellipsis for truncated rows
                        std::cout << " ... " << std::endl;
                        i = e.size() - 1u;
                    }
                    std::cout << ' ' << e[i] << ' ';

                }
                std::cout << std::endl;
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
                auto V_k = V.slice(Slice{0, V.nrows, 0, j}).transpose();
                auto z = V_k.mm(v);
                z = S_k.mm(z);
                z *= -tau;
                // copy T_01 = z
                S.copy(z, Slice{0, z.nrows, S_k.ncols, S_k.ncols + z.ncols});
                // copy T_11 = -tau
                S[S_k.nrows][S_k.ncols] = -tau;
            }
        }

        /*
        * ===============================================
        * Householder Reflection
        * -----------------------------------------------
        * Input: Column vector w
        * Output: w and w' (projection vector); tau (scaling factor)
        * ===============================================
        */
        template<typename T>
        Reflection<T> householder( Matrix<T> w) {

            assert( w.ncols == 1 && "Householder applied to column vectors only." );

            auto x_vec = w.col_slice(0, 0, w.nrows);
            auto s = -std::copysign(1, x_vec[0]);
            auto norm_x = norm(x_vec);

            // w = ( x − s||x||e1 ) / u1
            auto u1 = x_vec[0] - s * norm_x;
            w *= 1./u1;
            w[0][0] = 1.;
            T tau = -s * u1 / norm_x;

            // compute w'
            auto w_T = w.transpose();

            Reflection<T> result = { w, w_T, tau };
            return result;

        }

        /*
        * ===============================================
        * Householder Transformation
        * -----------------------------------------------
        * Input: Householder Reflection
        * Output: Transformation matrix
        * ===============================================
        */
        template<typename T>
        Matrix<T> hh_transform( Reflection<T> H ) {

            // Compute H = I - tau * w * w^T
            auto transform = H.w.mm(H.w_T);
            transform *= -H.tau;

            // subtract from I along diagonal
            for (auto d = 0u; d < transform.ncols; ++d) {
                transform[d][d] = 1 + transform[d][d];
            }
            return transform;
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


        void qr( Matrix <float>& A, Matrix <float>& S, Matrix <float>& V ) {

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
                R -= V.mm(Y);

                // compute Householder H_j from column A_j
                auto A_j = R.slice(j, m, j, j + 1);
                auto H_j = householder(A_j);

                // Compute Y(i+1:n,i): y_i = tau * R'v
                auto A_trail = R.slice(j, m, j, n).transpose();
                // matmul( A_trail, H_j.w, y, 0, 0, H_j.tau);
                y = A_trail.mm(H_j.w);
                y *= H_j.tau;


                // Update matrices V, Y
                V.copy(H_j.w, Slice{j, m, j, j + 1});
                Y.copy(y.transpose(), Slice{j, j + 1, j, n});

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(j), H_j.tau, S, V );

            }

            // update matrix with A <- A - VY'
            A -= V.mm(Y);

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


        void lq( Matrix <float>& A, Matrix <float>& S, Matrix <float>& U ) {

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
                result = X.mm(U);
                L -= result;

                // compute Householder H_i to eliminate right of diagonal
                auto A_i = L.slice(i, i + 1, i, n);
                auto H_i = householder(A_i.transpose());

                // x = tau * L'u
                auto A_trail = L.slice(i, m, i, n);
                //matmul( A_trail, H_i.w, x, 0, 0, H_i.tau);
                x = A_trail.mm(H_i.w);
                x *= H_i.tau;

                // Update matrices U, X
                X.copy(x, Slice{i, m, i, i + 1});
                U.copy(H_i.w_T, Slice{i, i + 1, i, n});
                U_T = U.transpose();

                // Store Housholder reflectors in compact form
                hholder_compact( size_t(i), H_i.tau, S, U_T);

            }
            // Update matrix: A <- A - XU'
            //matmul( X, U, result, 0, 0);
            //A -= result;
            A -= X.mm(U);

        }


        /*
        * ===============================================
        * Host Wrappers: Apply Householder for QR/LQ Decompositions
        * (Used for comparative testing only)
        * -----------------------------------------------
        * Subroutine applies S and V to matrix A
        * of a matrix A of size m x n for QR decomposition
        * Input:
        *  - A <Matrix> : m x n input matrix
        *  - S <Matrix> : m x m orthogonal matrix
        *  - V <Matrix> : m x m orthogonal matrix
        * ===============================================
        */

        void lq_apply( Matrix <float>& A, Matrix <float>& S, Matrix <float>& V ) {

            // Compute A <- A + AV'SV from compacted householders
            auto V_T = V.transpose();
            auto SV = S.mm(V);
            auto P = V_T.mm(SV);
            A += A.mm(P);
        }

        void qr_apply( Matrix <float>& A, Matrix <float>& S, Matrix <float>& V ) {

            // Compute A <- A + VSV'A from compacted householders
            auto V_T = V.transpose();
            auto SV_T = S.mm(V_T);
            auto Q = V.mm(SV_T);
            Q = Q.transpose();
            A += Q.mm(A);
        }


        /*
        * ===============================================
        * Blocked Band Reduction <brd_p1>
        * -----------------------------------------------
         * Dense matrix -> Banded matrix (Stage I of two-stage process)
         * -----------------------------------------------
         * Computes banded bidiagonal matrix B = U1'*A*V1 using
         * QR and LQ transformations to upper and lower diagonals
         * Input: Matrix <float> A (m x n matrix)
         * Output:
         *  - Matrix <float> B (banded bidiagonal m x n matrix)
         * ===============================================
         */

        Matrix<float> brd_p1( Matrix <float>& A, size_t const b_size ) {

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
                qr( A_panel_qr, S, V );

                // Update trailing matrix QA = (I − V T V')' A
                qr_apply( A_trail, S, V );

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
                    lq( A_panel_lq, S, U );

                    // Update trailing matrix AQ' =  A (I − V T V')'
                    lq_apply( A_trail, S, U );

                    // Copy LQ reduced panel to A
                    A.copy( A_panel_lq, Slice{ k, k + b_size, k + b_size, n} );

                }
                // A <- residual trailing matrix
                A.copy( A_trail, Slice{ m - A_trail.nrows, m, n - A_trail.ncols, n } );
            }
            return A;
        }

        /*
         * ===============================================
         * Bidiagonal Reduction (Golub-Kahan Algorithm) <brd>
         * -----------------------------------------------
         * Computes bidiagonal matrix B = U1'*A*V1 using
         * Householder transformations to upper/lower triangles
         * Input: Matrix <T> A (m x n matrix)
         * Output:
         *  - Matrix <T> B (bidiagonal m x n matrix)
         *  - Matrix <T> U1 (left-side orthogonal matrix)
         *  - Matrix <T> V1 (left-side orthogonal matrix)
         * ===============================================
         */
        template<typename T>
        Matrix<T> brd( Matrix <T>& A ) {

            Slice tgt;
            size_t b_size = 8u;

            // Iterate over blocks of A (size: b_size)
            for (auto k = 0u; k < A.ncols; k += b_size) {

                // Extract block
                auto A_blk = A.slice( k, A.nrows, k, A.ncols );
                auto m = A_blk.nrows;
                auto n = A_blk.ncols;

                // Initialize compact householder matrices
                auto V = Matrix<T>(m, b_size);
                auto Y = Matrix<T>(b_size, n);
                auto X = Matrix<T>(m, b_size);
                auto U = Matrix<T>(b_size, n);

                // Bidiagonalize block
                for (auto j = 0u; j < b_size; ++j) {

                    auto y = Matrix<T>(1, n); // 1 x n
                    auto x = Matrix<T>(m, 1); // m x 1

                    // (Step 1) Diagonal Reduction
                    // ---------------------------
                    // Reduce column j of A_(j−1)

                    //  A_(j-1) <- (A - VY' - XU')
                    auto A_diag = A_blk;
                    A_diag -= V.mm(Y);
                    auto XU = X.mm(U);
                    A_diag -= XU;

                    // compute Householder H_j from A_j to eliminate below diagonal
                    auto A_diag_j = A_diag.slice(j, m, j, j + 1);
                    auto H = householder(A_diag_j);

                    // Compute Y(i+1:n,i): y_i = tau * A_diag'v
                    auto A_trail = A_diag.slice(j, m, j, n).transpose();
                    y = A_trail.mm(H.w);
                    y *= H.tau;

                    // Update matrices V, Y
                    tgt = {j, m, j, j + 1};
                    V.copy(H.w, tgt);
                    tgt = {j, j+1, j, n};
                    Y.copy(y.transpose(), tgt);


                    // (Step 2) Super-diagonal Reduction
                    // ---------------------------
                    // compute row j of H_j.A_(j−1)

                    if (j < n - 1) {

                        // A_super <- A - VY' - XU'
                        auto A_super = A_blk;
                        A_super -= V.mm(Y);
                        A_super -= XU;

                        // compute Householder H_j to eliminate right of super-diagonal
                        auto A_super_j = A_super.slice(j, j + 1, j + 1, n);
                        H = householder(A_super_j.transpose());

                        // x = tau * A_super'u
                        auto A_trail_up = A_super.slice(j, m, j + 1, n);
                        x = A_trail_up.mm(H.w);
                        x *= H.tau;


                        // Update matrices X, U
                        tgt = {j, m, j, j + 1};
                        X.copy(x, tgt);
                        tgt = {j, j + 1, j + 1, n};
                        U.copy(H.w_T, tgt);

                    }
                }

                // Apply compact WY bidiagonalization to block
                // A[k:m, k:n] <- A[k:m, k:n] - VY' - XU'

                A_blk -= V.mm(Y);
                A_blk -= X.mm(U);

                // Update A
                tgt = {k, k + A_blk.nrows, k, k + A_blk.ncols};
                A.copy( A_blk, tgt );

            }

            //Bidiagonal<T> B = { A.diag(), A.diag(1) };
            return A;
        }

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
            auto x_T = x.transpose();
            auto H = householder(x_T);
            auto transform = hh_transform( H );
            A_t = A_t.mm(transform);
            A.copy( A_t, t );

            // Reduce col i + 1 below diagonal
            // {w, tau} = householder( A[i:i+shift,i+1:i+2] )
            auto end_j  = std::min( i + b_size + b_size - 1, A.ncols );
            t = Slice{ t.i1 + 1, t.i2, i + 1, end_j };
            A_t = A.slice( t );
            x = A_t.slice(0, A_t.nrows, 0, 1);
            H = householder(x);
            transform = hh_transform( H );
            A_t = transform.mm(A_t);
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
            auto H = householder(x.transpose());
            auto transform = hh_transform( H );
            A_t = A_t.mm(transform);
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
            auto H = householder(x);
            auto transform = hh_transform( H );
            A_t = transform.mm(A_t);
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
        Matrix<T> brd_p2( Matrix <T>& A, size_t const b_size = 0u ) {
            auto m = A.nrows;
            auto n = A.ncols;
            Slice t_left = {0,0,0,0};
            Slice t_right = {0,0,0,0};
            size_t end_i, end_j, start_j, end_j3;

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
            return A;
        }
    } // namespace gpu
} // namespace csc586
#endif // CS586_GPUSVD

