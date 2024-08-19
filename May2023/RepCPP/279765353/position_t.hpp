#pragma once

#include <cstddef> 

namespace nlohmann
{
namespace detail
{
struct position_t
{
std::size_t chars_read_total = 0;
std::size_t chars_read_current_line = 0;
std::size_t lines_read = 0;

constexpr operator size_t() const
{
return chars_read_total;
}
};

} 
} 
