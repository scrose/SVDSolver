

#ifndef CS586_SVD
#define CS586_SVD

/**
* **********************************************
* Singular Value Decomposition (Single-core Model)
 * Final Project
* **********************************************
 * CSC 586B - Spring 2020
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
*  - Rotation{}: Givens rotation
 *
* Input: Matrix (n x m)
* Output: Bidiagonal matrix as diagonal and super-diagonal vectors (n)
 *
* Functions:
*  - brd(): Golub-Kahan bidiagonal reduction
*  - diag_reduce(): Golub-Kahan QR diagonal reduction
*  - block_brd(): Optimized block-based bidiagonal reduction
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
#include "matrix.h"  // matrix class with operators

namespace csc586 {
    namespace serial {


        // Data type for Householder reflector as projection vector w and scalar tau
        template <typename T>
        struct Reflection {
            Matrix<T> w; // Householder vector
            Matrix<T> transform; // projection: I - tau.w.w^T
            T tau; // scalar normalizer
        };

        // Data type for parameters of Givens rotation (for angle theta)
        struct Rotation {
            typedef float T;
            T c;       // cosine(theta)
            T s;       // sine(theta)
            T r;       // radius
        };

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
        * SVD convergence criteria
        * See discussion in Demmel and Kahan, 1990.
        * -----------------------------------------------
        * get_tolerance(): returns tolerance t such that any
        * |e{i]| < t is set to zero during SVD diagonalization.
        * ===============================================
        */
        template <typename T>
        struct Criteria {
            T eps = 1e-8; // machine precision
            T umin = 1e-10; // underflow threshold ( smallest positive normalized number)
            T tolerance = 100*eps; // relative error tolerance
            size_t max_iter = 0; // maximum number of QR inner loops
            T threshold = 0; // threshold value

            // Calculate convergence threshold for e (see: Demmel and Kahan 1990, p.20)
            void init(const std::vector<T> d, const std::vector<T> e) {
                auto n = d.size();
                std::vector<T>lambda = std::vector<T>(n, 0);
                std::vector<T>mu = std::vector<T>(n, 0);
                // compute minimum sigma
                lambda.back() = std::abs(d.back());
                for (auto j = n - 1; j--;) {
                    lambda[j] = std::abs(d.at(j)) * lambda.at(j + 1) / (lambda.at(j + 1) + std::abs(e.at(j)));
                }

                mu.front() = std::abs(d.front());
                for (auto j = 0u; j < n - 1; ++j) {
                    mu[j + 1] = std::abs(d[j + 1]) * mu[j] / ( mu[j] + std::abs(e[j]));
                }
                auto lbound = std::min(
                        *std::min_element(lambda.begin(), lambda.end()),
                        *std::min_element(mu.begin(), mu.end())
                );
                max_iter = 500*n^2;
                threshold = std::max( tolerance*lbound, max_iter*umin);
            }
            // print out criteria
            void print()
            {
                std::cout << std::fixed;
                std::cout << std::setprecision(12);
                std::cout << "\nMax Iter\t" << max_iter << std::endl;
                std::cout << "Tolerance\t" << tolerance << std::endl;
                std::cout << "Threshold\t" << threshold << std::endl << std::endl;
            }

        };


        /*
        * ===============================================
        * Householder Reflection
        * -----------------------------------------------
        * Input: Column vector w
        * Output: x (projection vector); tau (scaling factor)
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

            // compute H = I - tau * w * w^T
            auto w_T = w.transpose();
            auto H = w.mm(w_T);
            H *= -tau;

            // subtract from I along diagonal
            for (auto d = 0u; d < H.ncols; ++d) {
                H[d][d] = 1 + H[d][d];
            }

            Reflection<T> result = { w, H, tau };
            return result;

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
        Bidiagonal<T> brd( Matrix <T>& A ) {

            auto m = A.nrows;
            auto n = A.ncols;
            Slice tgt;

            // Apply housholder reflections for each column j
            for (auto j = 0u; j < n; ++j) {

                // eliminate non-zeros to the left of the diagonal
                auto x = A.slice(j, m, j, j+1);
                // {w, tau} = householder( A[j:m,j:j+1] )
                auto H = householder(x);

                // extract trailing matrix [j:m,j:n] from A and update
                tgt = {j, m, j, n};
                auto minor = A.slice(tgt);
                A.copy( H.transform.mm(minor), tgt );

                // eliminate non-zeros to the right of super-diagonal using A^T
                if ( j < n - 1 ) {
                    // {w, tau} = householder( A’[j:j+1,j+1:n] )
                    x = A.slice(j, j+1, j+1, n);
                    H = householder(x.transpose());

                    // extract trailing matrix [j:m,j+1:n] from A and update
                    tgt = {j, m, j+1, n};
                    minor = A.slice(tgt);
                    A.copy( minor.mm(H.transform), tgt );

                }
            }
            Bidiagonal<T> B = { A.diag(), A.diag(1) };
            return B;
        }

        /*
        * ===============================================
        * Givens Rotation
        * -----------------------------------------------
        * Input: Vector elements [u1, u2]
        * Output: <Rotation> Givens rotation parameters
        * ===============================================
        */
        template<typename T>
        Rotation rotate(const T u1, const T u2) {
            T t1, t2, t3;
            Rotation params;

            if (u1 == 0)
            {
                params = {0.0, 1.0, u2};

            }
            else if (std::abs(u1) > std::abs(u2))
            {
                t1 = u2/u1, t2 = std::sqrt(1 + t1*t1), t3 = 1/t2;
                params = {t3, t1*t3, u1*t2};
            }
            else {
                t1 = u1/u2, t2 = std::sqrt(1 + t1*t1), t3 = 1/t2;
                params = {t1*t3, t3, u2*t2};
            }
            return params;
        }




        /*
         * ===============================================
         * Implicit Zero-Shift QR Algorithm
         * -----------------------------------------------
         * Adapted from "Accurate Singular Values of Bidiagonal Matrices" (Demmel, Kahan, 1990)
         * This algorithm begins and ends with vectors d[0] and d[1], representing
         * the diagonal and superdiagonal of a bidiagonal matrix. The vector d has length n.
         * Input: B (m x n bidiagonal matrix)
         * Output: Bidiagonal matrix as std::vector<T>
         * ===============================================
         */
        template<typename T>
        Bidiagonal<T> impl_zero_shift( Bidiagonal<T>& B ) {
            Rotation rot = {1.,0.,0.};
            Rotation rot_ = {1.,0.,0.};

            // sweep rotation along diagonal
            for (auto k = 0u; k < B.d.size() - 1; ++k) {

                rot = rotate( rot.c * B.d.at(k), B.e.at(k) );
                if (k > 0) {
                    B.e[k - 1] = rot.r * rot_.s;
                }
                rot_ = rotate(rot_.c * rot.r, B.d.at(k + 1) * rot.s);
                B.d[k] = rot_.r;

            }
            auto h = rot.c * B.d.back();
            B.e.back() = h * rot_.s;
            B.d.back() = h * rot_.c;
            return B;
        }


        /*
         * ===============================================
         * SVD - Fixed Iteration
         * -----------------------------------------------
         * Adapted from "Accurate Singular Values of Bidiagonal Matrices" (Demmel, Kahan, 1990)
         * This algorithm begins and ends with vectors d[0] and d[1], representing
         * the diagonal and superdiagonal of a bidiagonal matrix. The vector d has length n.
         * Input: B (m x n bidiagonal matrix)
         * Output: Sigma (SVD diagonal)
         * ===============================================
         */
        template<typename T>
        Bidiagonal<T> diag_reduce_fixed_iter( Bidiagonal<T>& B) {
            for (auto iter = 0u; iter < 200; ++iter) {
                impl_zero_shift(B);
            }
            return B;
        }

        /*
         * ===============================================
         * SVD - Convergent Application of "Chase-the-bulge" algorithm
         * QR Diagonalization
         * -----------------------------------------------
         * Adapted from "Accurate Singular Values of Bidiagonal Matrices" (Demmel, Kahan, 1990)
         * This algorithm begins and ends with vectors d[0] and d[1], representing
         * the diagonal and superdiagonal of a bidiagonal matrix. The vector d has length n.
         * Input: B (m x n bidiagonal matrix)
         * Output: Sigma (SVD diagonal)
         * ===============================================
         */
        template<typename T>
        Bidiagonal<T> qrd( Bidiagonal<T>& B) {

            // initialize convergence criteria
            Criteria<T> crit;
            crit.init(B.d, B.e);

            auto n = B.d.size();
            auto i_up = n - 2u;
            auto i_low = 0u;
            auto j = i_up;

            // iterate until convergence to threshold (limited by maximum iterations)
            for (auto iter = 0u; iter < crit.max_iter; ++iter) {

                // reduce problem size when zeros found on superdiagonal
                // (find bottommost nonscalar unreduced block diagonal submatrix of B)

                // zeros are near the bottom right
                for (auto i = i_up; i >= 1u; --i) {
                    i_up = i;
                    if (std::abs(B.e[i]) > crit.threshold) break;
                }

                // zeros are near the top left
                j = i_up;
                for (auto i = i_low; i < i_up; ++i) {
                    if (std::abs(B.e[i]) > crit.threshold) {
                        j = i;
                        break;
                    }
                }
                i_low = j;
                if ((i_up == i_low && std::abs(B.e[i_up]) <= crit.threshold) || (i_up < i_low))
                {
                    // SVD completed, sort absolute singular
                    std::transform(B.d.begin(), B.d.end(), B.d.begin(),
                                   [](const T e) -> T { return std::abs(e); });
                    std::sort(B.d.begin(), B.d.end(), std::greater<T>());
                    return B;
                }

                auto B_reduce = B.slice(i_low, i_up + 1, i_low, i_up);

                // do an implicit zero shift operation
                impl_zero_shift(B_reduce);

                // copy back to B
                std::copy(B_reduce.d.begin(), B_reduce.d.end(), B.d.begin() + i_low);
                std::copy(B_reduce.e.begin(), B_reduce.e.end(), B.e.begin() + i_low);

            }
            std::cout << "Error: Maximum iterations reached without convergence." << std::endl;
            return B;

        }



        /*
        * ===============================================
        * Blocked Bidiagonal Reduction <block_brd>
        * -----------------------------------------------
         * Computes bidiagonal matrix B = U1'*A*V1 using
         * Householder transformations to upper/lower triangles
         * Input:
         *  - Matrix <T> A: m x n matrix
         *  - <size_t> b_size: Block (panel) size
         * Output:
         *  - Matrix <T> B (bidiagonal m x n matrix)
         *  - Matrix <T> U1 (left-side orthogonal matrix)
         *  - Matrix <T> V1 (left-side orthogonal matrix)
        * ===============================================
        */
        template<typename T>
        Bidiagonal<T> block_brd( Matrix <T>& A, const size_t b_size ) {

            Slice tgt;

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
                        U.copy(H.w.transpose(), tgt);

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

            Bidiagonal<T> B = { A.diag(), A.diag(1) };
            return B;
        }


    } // namespace serial
} // namespace csc586
#endif // CS586_SVD
