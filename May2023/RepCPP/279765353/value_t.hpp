#pragma once

#include <array> 
#include <cstddef> 
#include <cstdint> 
#include <string> 

#include <nlohmann/detail/boolean_operators.hpp>

namespace nlohmann
{
namespace detail
{


enum class value_t : std::uint8_t
{
null,             
object,           
array,            
string,           
boolean,          
number_integer,   
number_unsigned,  
number_float,     
binary,           
discarded         
};


inline bool operator<(const value_t lhs, const value_t rhs) noexcept
{
static constexpr std::array<std::uint8_t, 9> order = {{
0 , 3 , 4 , 5 ,
1 , 2 , 2 , 2 ,
6 
}
};

const auto l_index = static_cast<std::size_t>(lhs);
const auto r_index = static_cast<std::size_t>(rhs);
return l_index < order.size() and r_index < order.size() and order[l_index] < order[r_index];
}
}  
}  
