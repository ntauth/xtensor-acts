/**
*  @author Ayoub Chouak (@ntauth)
*  @file   main.hpp
*  @brief  XTensor Code Samples for ACTS (C++14)
*
*/

#include <iostream>
#include <random>
#include <chrono>
#include <string>

#include <xtensor/xarray.hpp>

/*
* @brief Root namespace
*/
namespace xt_pg
{
    class output_section
    {
        protected:
            std::string name;
        public:
            explicit output_section(std::string const&);
            std::string get_name() const { return name; }
            void set_name(std::string const& name_) { name = name_; }

            friend std::ostream& operator <<(std::ostream&, output_section const&);
    };

    std::ostream& operator <<(std::ostream&, output_section const&);
}