/**
*  @author Ayoub Chouak (@ntauth)
*  @file   main.cpp
*  @brief  XTensor Code Samples
*          
*/

#include <main.hpp>
#include <utils.hpp>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

int main(int argc, char** argv)
{
    auto  sys_tm_start = std::chrono::system_clock::now();
    auto& xt_rnd_engine = xt::random::get_default_random_engine();
	unsigned long rnd_mat_idx;
    long x_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Test echelon reduction
    std::cout << xt_pg::output_section("ECHELON REDUCTION");

    xt::xarray<double> echelon_input = {
            { 0, 0, 0 },
            { 1, 0, 0 },
            { 0, 0, 3 }
    };

    xt_pg::xt_utils::reduce_echelon(echelon_input);
    std::cout << echelon_input << std::endl;

    // Seed the random engine and generate a random index
    xt_rnd_engine.seed(x_seed);
    rnd_mat_idx = xt_rnd_engine() % xt_pg::test_matrices.size();

    xt::xarray<double> rnd_mat = xt_pg::test_matrices[rnd_mat_idx];
	auto const& shape = rnd_mat.shape();

    /**
     * Test with the hardcoded matrices from test_matrices
     */
    std::cout << xt_pg::output_section("TEST MATRICES");

    xt::xarray<double> rnd_mat_inv;

    for (auto const& xa : xt_pg::test_matrices)
    {
        std::cout << "Matrix: " << std::endl;
        std::cout << rnd_mat << std::endl;

       rnd_mat_inv = rnd_mat;
        xt_pg::xt_utils::invert_gauss_jordan(rnd_mat_inv);

        // Print the inverse
        std::cout << "Inverse: " << std::endl;
        std::cout << rnd_mat_inv << std::endl;
    }

    /**
     * Test with a random-generated matrix
     */
    std::vector<size_t> rnd_shape;
    unsigned long rnd_dimension = xt_rnd_engine() % 5 + 1;
    double lub = 1e10, glb = -1e10;

    rnd_shape = { rnd_dimension, rnd_dimension };

    std::cout << xt_pg::output_section("RANDOM MATRICES");
    std::cout << "[+] Testing with a " << rnd_dimension << "x" << rnd_dimension << " random-generated matrix." << std::endl;
    rnd_mat = xt_pg::xt_utils::generate_random_xarray(rnd_shape, lub, glb);

    // Print the random-generated matrix
    std::cout << rnd_mat << std::endl;

    rnd_mat_inv = rnd_mat;
    xt_pg::xt_utils::invert_gauss_jordan(rnd_mat_inv);

    // Print the inverse
    std::cout << xt_pg::output_section("INVERSE");
    std::cout << rnd_mat_inv << std::endl;

    auto sys_tm_end = std::chrono::system_clock::now();

    std::cout << "[+] Program terminated in " << std::chrono::duration_cast<std::chrono::milliseconds>(sys_tm_end - sys_tm_start).count() << " ms";
}

xt_pg::output_section::output_section(std::string const& section_name)
{
    name = section_name;
}

std::ostream &xt_pg::operator<<(std::ostream &os, const xt_pg::output_section& o_sect)
{
    os << "\t\t---- " << o_sect.get_name() << " ----" << std::endl;
    return os;
}