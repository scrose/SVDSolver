//
// Created by Spencer Rose on 2020-03-11.
//

#ifndef CSC586C_SVD_MATRIX
#define CSC586C_SVD_MATRIX


#include <iostream>
#include <iomanip>
#include <cassert>		 // assert()
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <random>
#include <fstream>
#include <iterator>
#include <functional>
#include <omp.h>


/**
* **********************************************
* Singular Value Decomposition: Matrix Class
* **********************************************
 * CSC 586B - Spring 2020 - Project
 * Author: Spencer Rose
 * Class to store and operate on 2D mxn matrices
 *
* Data Structures:
*  - Matrix(): matrix diagonal and bidiagonal
*  - Slice{}: stores matrix indices for a slice
* **********************************************
**/



namespace csc586 {

    // Data type for indexing matrix slices
    struct Slice {
        size_t i1; // row start dimension
        size_t i2; // row end dimension
        size_t j1; // column start dimension
        size_t j2; // column end dimension
        // comparator to determine if slices
        bool contains( const Slice s)
        {
            return (s.i2 - s.i1 <= i2 - i1) && ( s.j2 - s.j1 <= j2 - j1);
        }
    };

    /* Matrix utility functions */

    // Get Euclidean normalization ||v|| for input vector v
    // Input : vector v / Output : ||v||
    /* NOTE: std::sqrt is required by the IEEE standard to be exact.
     * */
    template < typename T > T norm(std::vector<T>& v) {
        const T init_val = 0;
        return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), init_val));
    }


    /**
     * ***********************************************
     * Matrix Data Structure
     * * **********************************************
     * Defines a matrix of type T
     * Input : m = number of rows, n = number of columns, max_val = max element value
     * Output : ncols by nrows n matrix of integer values from 0 to max_val
     *
     * NOTE: In c++11, the std::vector methods erase and insert take const_iterators
     * instead of iterators, i.e. automatic conversion does not work.
     * **********************************************
     */


    template <typename T>
    class Matrix
    {
        std::vector< std::vector< T > > elements_;

    public:
        size_t nrows, ncols;
        bool parallel;

        // Constructors
        Matrix () {};
        // dimensional constructor
        Matrix ( const size_t& row_dim, const size_t& col_dim, const bool parallel = false )
        {
            this->parallel = parallel;
            if (parallel) {
                nrows = row_dim; // row dimension
                ncols = col_dim; // column dimension
                std::vector<T> row_v ( ncols, 0);
                elements_.resize( nrows, row_v );
            }
            else {
                nrows = row_dim; // row dimension
                ncols = col_dim; // column dimension
                std::vector<T> row_v ( ncols, 0);
                elements_.resize( nrows, row_v );
            }


        }
        // array constructor
        Matrix ( const T *arr, size_t nrows, size_t ncols)
        {
            this->nrows = nrows; // row dimension
            this->ncols = ncols; // column dimension
            for ( auto i = 0u; i < nrows; ++i )
            {
                std::vector<T> row_v(arr + i * ncols, arr + i * ncols + ncols);
                this->elements_.push_back(row_v);
            }
        }


        // [Operator] Overload subscript operator: returns row of matrix at index
        inline std::vector<T>& operator[] ( size_t idx ) { return elements_.at(idx); }

        // [Operator] Sum this matrix to matrix B and assign result to this matrix
        inline Matrix<T>& operator+= ( const Matrix<T>& m )
        {
            assert( nrows == m.nrows && "Matrix 1 row dim must match matrix 2 row dim." );
            assert( ncols == m.ncols && "Matrix 1 col dim must match matrix 2 col dim." );

            if (parallel) {
                for (auto i = 0u; i < nrows; ++i)
                    for (auto j = 0u; j < nrows; ++j)
                        elements_[i][j] += m.elements_[i][j];
            }
            else {

                for ( auto i = 0u; i < nrows; ++i ) {
                    std::transform(
                            m.elements_[i].begin(),
                            m.elements_[i].end(),
                            elements_[i].begin(),
                            elements_[i].begin(),
                            std::plus<T>()
                    );
                }
            }
            return *this;
        }

        // [Operator] Substract matrix B from this matrix and assign result to this matrix
        inline Matrix<T>& operator-= ( const Matrix<T>& m )
        {
            assert( nrows == m.nrows && "Matrix 1 row dim must match matrix 2 row dim." );
            assert( ncols == m.ncols && "Matrix 1 col dim must match matrix 2 col dim." );
            if (parallel) {
                for (auto i = 0u; i < nrows; ++i)
                    for (auto j = 0u; j < nrows; ++j)
                        elements_[i][j] -= m.elements_[i][j];
            }
            else {
                for ( auto i = 0u; i < nrows; ++i ) {
                    std::transform(
                            elements_[i].begin(),
                            elements_[i].end(),
                            m.elements_[i].begin(),
                            elements_[i].begin(),
                            std::minus<T>()
                    );
                }

            }
            return *this;
        }

        // [Operator] Multiply this matrix with scalar alpha and assign result to this matrix
        inline Matrix<T>& operator*= ( const T alpha )
        {
            if (parallel) {
                for (auto i = 0u; i < nrows; ++i)
                    for (auto j = 0u; j < nrows; ++j)
                        elements_[i][j] *= alpha;
            }
            else {
                std::for_each( elements_.begin(), elements_.end(),
                       [&alpha](std::vector<T>& row) {
                           std::transform(row.begin(), row.end(), row.begin(),
                                   std::bind(std::multiplies<T>(), std::placeholders::_1, alpha));
                       });
            }
            return *this;
        }



        /*
         * ===============================================
         * Matrix Operations
         * ===============================================
         * */

        // Returns size of matrix
        size_t size() const {
            return elements_.size() * elements_[0].size();
        }


        // Returns transposed matrix [i,j] -> [j,i]
        Matrix <T> transpose( const bool parallel = 0 ) {
            auto tmp = Matrix<T>(ncols, nrows);
            #pragma omp parallel for schedule (static) if ( parallel && (size() > 512) )
            for (auto i = 0u; i < nrows; ++i)
                for (auto j = 0u; j < ncols; ++j)
                    tmp[j][i] = elements_[i][j];
            return tmp;
        }


        /*
         * ===============================================
         * Matrix Multiplication (non-parallelized)
         * -----------------------------------------------
         * Transposed Multiplication of AB^T or AB to improve
         * sequential access of matrix M elements.
         * Input:
         *  - Matrix <T> A (m x n matrix)
         *  - Matrix <T> B (n x p matrix)
         * Output:
         *  - Matrix <T> AB (m x p matrix)
         *  OR:
         *  - Matrix <T> AB^T (m x p matrix)
         * ===============================================
         */
        Matrix <T> mm( Matrix& M, const bool parallel = 0 ) const {

            assert( ncols == M.nrows && "Matrix 1 col dim must match Matrix 2 row dim." );

            auto result = Matrix <T> ( nrows, M.ncols );
            Matrix <T> tmp;
            tmp = M.transpose();

            #pragma omp parallel  for schedule (static) if ( parallel && (size() > 256 || M.size() > 256 ) )
            for( auto i = 0u; i < nrows; ++i )
                for( auto j = 0u; j < M.ncols; ++j )
                    for( auto k = 0u; k < ncols; ++k)
                        result[i][j] += elements_[i][k] * tmp[j][k];
            return result;
        }


        /*
         * ===============================================
         * Flatten mattrix to 1D
         * -----------------------------------------------
         * Returns 1-D matrix of concatenated rows
         * ===============================================
         */
        Matrix<T> flatten( const bool transpose = 0 ) const {
            auto tmp = Matrix<T>( 1, size() );
            if ( transpose ) {
                for( auto i = 0u; i < nrows; ++i )
                    for( auto j = 0u; j < ncols; ++j )
                        tmp[0][j*nrows + i] = elements_[i][j];
            }
            else {
                for( auto i = 0u; i < nrows; ++i )
                    for( auto j = 0u; j < ncols; ++j )
                        tmp[0][i*ncols + j] = elements_[i][j];
            }
            return tmp;
        }




        // Copy slice of elements from src matrix -> tgt matrix (src slice to tgt slice)
        void copy( Matrix <T> src, Slice s, Slice t ) {
            assert( t.contains(s) && "Slice range from source outside target range." );
            auto i_tgt = t.i1;
            for ( auto i_src = s.i1; i_src < s.i2; ++i_src, ++i_tgt) {
                std::copy(
                        src.elements_[i_src].begin() + s.j1,
                        src.elements_[i_src].end() - (src.ncols - s.j2),
                        elements_[i_tgt].begin() + t.j1
                );
            }
        }

        // [Overloaded] Copy all elements from src matrix -> tgt matrix (to tgt slice)
        void copy( Matrix <T> src, Slice t ) {

            assert( t.i2 - t.i1 <= nrows && t.j2 - t.j1 <= ncols && "Copy range from source outside target size." );
            assert( t.i2 - t.i1 <= src.nrows && t.j2 - t.j1 <= src.ncols && "Copy target range be in source range." );

            auto i_tgt = t.i1;
            for ( auto i_src = 0u; i_src < src.nrows; ++i_src, ++i_tgt ) {
                std::copy(
                        src.elements_[i_src].begin(),
                        src.elements_[i_src].end(),
                        elements_[i_tgt].begin() + t.j1
                );
            }
        }


        // [Overloaded] Copy all elements from src matrix -> tgt matrix at front
        void copy( Matrix <T> src ) {
            auto i_tgt = 0u;
            assert( src.nrows <= nrows && src.ncols <= ncols && "Copy range for source outside target range." );
            for ( auto i_src = 0u; i_src < src.nrows; ++i_src, ++i_tgt) {
                std::copy( src.elements_[i_src].begin(), src.elements_[i_src].end(), elements_[i_tgt].begin());
            }
        }

        // Concatenates matrix B row-wise below this matrix
        void row_concat( Matrix<T> B ) {
            assert( B.ncols == ncols && "Column dimensions must match for row concatenation." );
            elements_.insert( elements_.end(), B.elements_.cbegin(), B.elements_.cend() );
            // Update row size
            nrows = elements_.size();
        }

        // Concatenates matrix B column-wise right of this matrix
        void col_concat( Matrix<T> B ) {
            assert( B.nrows == nrows && "Row dimensions must match for column concatenation." );
            for ( auto i = 0u; i < nrows; ++i) {
                elements_[i].insert( elements_[i].end(), B.elements_[i].begin(), B.elements_[i].end());
            }
            // Update col size
            ncols = elements_[0].size();
        }

        // Fill matrix with scalar value at target slice
        void fill( const T value, const Slice t ) {
            assert( t.i2 - t.i1 <= nrows && t.j2 - t.j1 <= ncols && "Copy range for source outside target range." );
            std::for_each(elements_.begin(), elements_.end(), [=](std::vector<T> &row){
                std::fill( row.begin() + t.i1, row.end() - (ncols - t.j2), value);
            });
        }

        // Resize matrix to size of slice
        void resize( const size_t m, const size_t n ) {
            nrows = m;
            ncols = n;
        }


        // [Overloaded] Fill matrix with random values in given range
        // Input : min_val (Minimum range value), max_val (Maximum range value)
        void fill( const T& min_val, const T& max_val ) {
            for ( auto &row : elements_ ) {
                std::generate(
                        row.begin(),
                        row.end(),
                        [=]() {
                            std::random_device rd;  // obtain seed for the random number engine
                            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
                            std::uniform_real_distribution<> dis( min_val, max_val );
                            return static_cast <T> ( dis(gen) );
                        }
                );
            }
        }

        // Returns matrix diagonal as vector; offset can access superdiagonals/subdiagonals
        std::vector <T> diag(size_t offset=0) {
            auto tmp = std::vector<T>( ncols - offset, 0 );
            auto i = 0u;
            std::for_each(elements_.begin(), elements_.end() - offset, [&](auto &v){ tmp[i] = v[i + offset]; ++i; });
            return tmp;
        }



        // Returns matrix slice
        Matrix <T> slice( const size_t row_start, const size_t row_end, const size_t col_start, const size_t col_end ) const {
            size_t j = 0u;
            auto tmp = Matrix <T> ( row_end - row_start, col_end - col_start );
            for( auto i = row_start; i < row_end; ++i, ++j )
                std::copy(
                        elements_[i].begin() + col_start,
                        elements_[i].end() - (ncols - col_end),
                        tmp.elements_[j].begin()
                );
            return tmp;
        }



        // [Overloaded] Returns matrix slice
        Matrix <T> slice( const Slice& s ) const {
            size_t j = 0u;
            auto tmp = Matrix <T> ( s.i2 - s.i1, s.j2 - s.j1 );
            for( auto i = s.i1; i < s.i2; ++i, ++j )
                std::copy(
                        elements_[i].begin() + s.j1,
                        elements_[i].end() - (ncols - s.j2),
                        tmp.elements_[j].begin()
                );
            return tmp;
        }



        // Extract tile from matrix
        Matrix<T> get_tile( const size_t i, const size_t j, const size_t nbt) {
            auto t_size = size_t ( nrows / nbt );
            auto x1 = i*t_size;
            auto y1 = j*t_size;
            auto x2 = x1 + t_size;
            auto y2 = y1 + t_size;
            assert( x2  <= nrows && y2 <= ncols && "Tile out of range of matrix." );
            return slice(Slice{ x1, x2, y1, y2});
        }


        // Copy tile to matrix
        void set_tile( Matrix<T> tile, const size_t i, const size_t j, const size_t nbt) {
            auto t_size = size_t ( nrows / nbt );
            auto x1 = i*t_size;
            auto y1 = j*t_size;
            auto x2 = x1 + t_size;
            auto y2 = y1 + t_size;
            assert( x2  <= nrows && y2 <= ncols && "Tile out of range of matrix." );
            copy(tile, Slice{ x1, x2, y1, y2});
        }

        // [Overloaded] Fill tile with value
        void set_tile( const T value, const size_t i, const size_t j, const size_t nbt) {
            auto t_size = size_t ( nrows / nbt );
            auto x1 = i*t_size;
            auto y1 = j*t_size;
            auto x2 = x1 + t_size;
            auto y2 = y1 + t_size;
            assert( x2  <= nrows && y2 <= ncols && "Tile out of range of matrix." );
            this->fill(value, Slice{ x1, x2, y1, y2});
        }


        // Returns extracted column from matrix
        std::vector<T> col_slice( size_t j, size_t row_start, size_t row_end ) {
            assert( row_end > row_start && "Slice start must be less than slice end." );
            auto tmp = std::vector<T> ( row_end - row_start, 0);
            auto k = 0u;
            for( auto i = row_start; i < row_end; ++i, ++k)
                tmp[k] = elements_[i][j];
            return tmp;
        }


        /*
         * ===============================================
         * I/O Operations
         * ===============================================
         * */

        // Writes matrix to file
        void write( std::string const& filepath ) const {
            std::cout << filepath << std::endl;
            // create output file stream
            std::ofstream data_file;
            data_file.open( filepath, std::ios::out | std::ios::app | std::ios::binary );

            if (data_file.fail())
                std::cout << "Error writing to file." << std::endl;

            for (auto i = 0u; i < nrows; ++i)
            {
                for (auto j = 0u; j < ncols; ++j) {
                    auto value = elements_[i][j];
                    data_file.write(reinterpret_cast<char *>(&value), sizeof(T));
                }
            }
            data_file.close();
        }

        // Reads matrix to file (must match dimensions)
        void read ( std::string const& filepath ) {
            std::ifstream data_file;
            data_file.open( filepath, std::ios::in | std::ios::binary);
            for (int i = 0u; i < nrows; ++i)
            {
                for (int j = 0u; j < ncols; ++j) {
                    data_file.read(reinterpret_cast<char*>(&elements_[i][j]), sizeof(float));
                }
            }
            data_file.close();
        }



        // Prints matrix to console
        void print( const uint32_t& truc = 16u ) const {
            std::cout << std::fixed;
            std::cout << std::setprecision(6);
            auto overhead = sizeof(Matrix) + ( nrows + 1)*sizeof(std::vector<T>);
            auto payload = sizeof(T) * size();

            // Matrix memory profile information
            std::cout << "\n-------\nMatrix capacity: " << elements_.capacity();
            std::cout << " [" << size() << " elements; m = " << nrows << ", n = " << ncols << "]" << std::endl;
            std::cout << "Matrix overhead: " << overhead << 'b' << std::endl;
            std::cout << "Size of Payload: " << payload << 'b' << std::endl;
            std::cout << "Matrix total size: " << overhead + payload << 'b' << std::endl;
            std::cout << std::endl;
            for( auto i = 0u; i <= truc && i < nrows; ++i )
            { // iterate rows
                if ( i == truc ) { // add ellipsis for truncated rows
                    std::cout << " ... " << std::endl;
                    i = nrows - 1u;
                }
                for( auto j = 0u; j <= truc && j < ncols; ++j )
                { // iterate columns
                    if ( j == truc )
                    { // add ellipsis for truncated cols
                        std::cout << "... ";
                        j = ncols - 1u;
                    }
                    std::cout << ' ' << elements_[i][j] << ' ';

                }
                std::cout << std::endl;
            }
        }

    };

} // namespace csc586


#endif //CSC586C_SVD_MATRIX_ALIGNED_H
