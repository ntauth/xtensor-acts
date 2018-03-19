/**
*  @author Ayoub Chouak (@ntauth)
*  @file   utils.hpp
*  @brief  Utilities for XTensor
*
*/

#pragma once

#include <climits>
#include <chrono>
#include <vector>
#include <iostream>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xexception.hpp>

namespace xt_pg
{

    //!< Hardcoded NxN matrices for testing
    const std::vector<xt::xarray<double>> test_matrices =
    {
            {
                { 4, 7.5 },
                { 3.0, 13.799 }
            },
            {
                { 4, 7.5, 13.244 },
                { 3.0, 13.799, 1.009 },
                { 4.7398, 140.1, 37.0001 }
            },
            {
                { 4, 7.5, 13.244, 5 },
                { 3.0, 13.799, 1.009, 42 },
                { 4.7398, 140.1, 37.0001, 399 },
                { 4, 7.5, 13.244, 24 },
                { 16, 29.1, 44, 7 }
            },
            {
                { 2,  11, 3,  9,  4  },
                { 5,  10, 12, 13, 14 },
                { 6,  8,  15, 16, 7  },
                { 17, 18, 19, 20, 21 },
                { 22, 23, 24, 25, 26 }
            }
    };

    //!< Floating point accuracy error
    constexpr double cx_fp_bias = 1e-18;

    //!< Helper for creating verbose exception messages
    inline std::string exception_verbose(std::string const& what) {
        return what + " - Exception thrown at " + std::string(__FILE__) + ": " + std::string(__FUNCTION__) + "@" + std::to_string(__LINE__);
    }

    class xt_utils
    {
        public:
            /**
             *
             * @tparam Ty type of the matrix elements
             * @param mat0 the matrix whose row is to be swapped
             * @param mat1 the matrix whose row is to be swapped
             * @param row0 the index of the row of mat0 that is to be swapped
             * @param row1 the index of the row of mat1 that is to be swapped
             */
            template<typename Ty>
            static void swap_row(xt::xarray<Ty>& mat0, xt::xarray<Ty>& mat1, size_t row0, size_t row1);
            /**
             * @brief Reduces a matrix to the row echelon form
             * @tparam Ty matrix element type
             * @param mat the matrix to be reduced
             */
            template<typename Ty>
            static void reduce_echelon(xt::xarray<Ty>& mat);
            /**
             * @brief Inverts a matrix using the Gauss-Jordan method
             * @tparam Ty matrix element type
             * @param mat the matrix to be reduced
             */
            template<typename Ty>
            static void invert_gauss_jordan(xt::xarray<Ty>& mat);
            /**
             * @brief Generates a random matrix given the bounds and the shape
             * @tparam Ty type of matrix elements
             * @tparam S  type of shape
             * @param shape the shape of the matrix to be generated
             * @param lub   the lowest upper bound of the matrix to be generated
             * @param  glb  the greatest lower bound of the matrix to be generated
             * @return
             */
            template<typename Ty, typename S = std::vector<size_t>>
            static xt::xarray<Ty> generate_random_xarray(S const& shape, Ty lub, Ty glb);
    };

    template<typename Ty>
    inline void xt_utils::invert_gauss_jordan(xt::xarray<Ty>& mat)
    {
        xt::xarray<Ty> mat_;

        // Get the matrix shape
        auto const& shape = mat.shape();

        // Augment the matrix by adjoining the identity matrix to the right
        std::vector<size_t> am_shape = { shape[0], 2 * shape[0] };
        mat_ = xt::xarray<Ty>(am_shape);

        // Fill the original matrix
        for (size_t i = 0; i < am_shape[0]; i++)
            for (size_t j = 0; j < am_shape[0]; j++)
                mat_(i, j) = mat(i, j);

        // Fill the identity matrix
        for (size_t i = 0; i < am_shape[0]; i++)
            for (size_t j = am_shape[0]; j < am_shape[1]; j++)
                mat_(i, j) = (j - am_shape[0] == i) ? 1 : 0;

        // Reduce the augmented matrix to echelon form and extract the inverse
        reduce_echelon(mat_);

        // Copy the inverse to mat
        for (size_t i = 0; i < am_shape[0]; i++)
        {
            for (size_t j = am_shape[0]; j < am_shape[1]; j++) {
                mat(i, j - am_shape[0]) = mat_(i, j);
            }
        }
    }

    template<typename Ty>
    inline void xt_utils::reduce_echelon(xt::xarray<Ty>& mat)
    {
        auto const& shape = mat.shape();
        size_t last_nz_idx = std::numeric_limits<size_t>::max();

        for (size_t col = 0; col < shape[1]; col++)
        {
            size_t nz_idx = std::numeric_limits<size_t>::max();
            bool  nz_found = false;

            for (size_t row = col; row < shape[0] && !nz_found; row++)
            {
                if (mat(row, col) != 0)
                {
                    nz_idx = row;
                    nz_found = true;
                }
            }

            if (nz_found)
            {
                // Move the row to allow for the zero rows to cascade down
                if (nz_idx != col)
                    swap_row(mat, mat, nz_idx, col);


                Ty pivot = mat(col, col);

                // Reduce the row to echelon form by dividing each element by pivot
                for (size_t i = 0; i < shape[1]; i++)
                {
                    mat(nz_idx, i) /= pivot;
                }

                // Apply a gauss move to zero out the elements in the rows above and below
                for (size_t i = 0; i < shape[0]; i++)
                {
                    if (i != col && mat(i, col) != 0)
                    {
                        Ty lambda = mat(i, col); // Reduction factor

                        for (size_t j = 0; j < shape[1]; j++)
                            mat(i, j) -= mat(col, j) * lambda;
                    }
                }
            }
        }
    }

    template<typename Ty, typename S = std::vector<size_t>>
    inline xt::xarray<Ty> xt_utils::generate_random_xarray(S const& shape, Ty lub, Ty glb)
    {
        auto& rnd_engine = xt::random::get_default_random_engine();
        long x_seed = std::chrono::system_clock::now().time_since_epoch().count();

        // Seed xtensor's random engine
        rnd_engine.seed(static_cast<unsigned long>(x_seed));

        // Generate a random matrix
        xt::xarray<Ty> rnd_mat = xt::random::rand(shape, glb, lub, rnd_engine);

        return rnd_mat;
    }

    template<typename Ty>
    inline void xt_utils::swap_row(xt::xarray<Ty>& mat0, xt::xarray<Ty>& mat1, size_t row0, size_t row1)
    {
        xt::xarray<Ty> x_view0 = xt::view(mat0, row0);
        xt::xarray<Ty> x_view1 = xt::view(mat1, row1);
        auto const& shape0 = mat0.shape();
        auto const& shape1 = mat1.shape();

        // Make sure the rows are not out of bound
        xt::check_access(shape0, row0);
        xt::check_access(shape1, row1);

        // Make sure the row dimensions are compatible
        if (shape0[0] != shape1[0])
            throw std::runtime_error(exception_verbose("Incompatible row dimension!"));

        for (size_t i = 0; i < x_view0.size(); i++) {
            mat0(row0, i) = x_view1(i);
            mat1(row1, i) = x_view0(i);
        }
    }
}