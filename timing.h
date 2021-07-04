/**
 * Timing library to benchmark and comparatively analyse different implementation approaches
 */

#ifndef CS586_TIMING
#define CS586_TIMING

#include <algorithm> // std::for_each()
#include <chrono>	 // timing libraries
#include "matrix.h"  // matrix class with operators
#include "svd_serial.h"  // matrix class with operators

namespace csc586 {
    namespace benchmark {

        using duration = float;

/**
 * Benchmarks the average time to run Function f, using the provided container of test_instances.
 * For more accurate timing results, it is best to provide many independently generated test instances.
 */

        template < typename Callable, typename Container >
        duration benchmark( Callable f, Container test_instances )
        {

            //using output_type = decltype( f( test_instances.front() ) );

            //output_type output;
            csc586::serial::Bidiagonal<float> output = {};

            // starts the timer. We use the chrono library not just because it is the idiomatic approach,
            // but also because it offers precision to the nanosecond, if we want it.
            auto const start_time = std::chrono::steady_clock::now();

            std::for_each( std::cbegin( test_instances )
                    , std::cend  ( test_instances )
                    , [&output, f]( auto x ){ output = f(x); } );

            // end timer
            auto const end_time = std::chrono::steady_clock::now();

            // do something arbitrary with output. In this case, we print it out.
            //std::cout << output << std::endl;
            //output.print();

            // return average time
            // the syntax for this library is a bit cumbersome...
            return std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count()
                   / static_cast< duration >( test_instances.size() );
        }


        template < typename Callable, typename Container >
        duration benchmark( Callable f, Container test_instances, size_t const b_size )
        {

            using output_type = decltype( f( test_instances.front(), b_size) );
            output_type output;

            //output_type output;
            //serial::Bidiagonal<float> output = {};

            // starts the timer. We use the chrono library not just because it is the idiomatic approach,
            // but also because it offers precision to the nanosecond, if we want it.
            auto elapsed_time = std::chrono::steady_clock::duration::zero();

            // run function f on every random test instance, arbitrarily summing the return values.
            // Note that for_each is a standard library function that applies a functor (the third argument)
            // to everything in the range given by the first two parameters (start and end).
            //
            // cbegin() returns an iterator to the first element in a container (such as a vector or array).
            // In particular, the iterator is *const*; i.e., we cannot modify the contents using the iterator.
            // cend() returns an iterator one past the last element. So looping from cbegin() to cend() will
            // iterate everything in test_instances.
            std::for_each( std::cbegin( test_instances )
                    , std::cend  ( test_instances )
                    , [&output, f, b_size, &elapsed_time]( auto x ){
                        auto const start_time = std::chrono::steady_clock::now();
                        output = f(x, b_size);
                        auto const end_time = std::chrono::steady_clock::now();
                        elapsed_time += end_time - start_time;
            } );

            // do something arbitrary with output. In this case, we print it out.
            //output.print(8);

            // return average time
            return std::chrono::duration_cast<std::chrono::microseconds>( elapsed_time ).count()
                   / static_cast< duration >( test_instances.size() );
        }

        // Print elapsed time to std::out
        long long int calc_time(
                std::chrono::system_clock::time_point start_time,
                std::chrono::system_clock::time_point end_time) {
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return elapsed_time.count();
        }


    } // namespace benchmark
} // namespace csc586

#endif // CS586_TIMING
