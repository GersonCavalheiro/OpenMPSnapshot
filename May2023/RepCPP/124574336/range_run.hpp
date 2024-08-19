
#if !defined(BOOST_SPIRIT_RANGE_RUN_MAY_16_2006_0801_PM)
#define BOOST_SPIRIT_RANGE_RUN_MAY_16_2006_0801_PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/char_set/range.hpp>
#include <vector>

namespace boost { namespace spirit { namespace support { namespace detail
{
template <typename Char>
class range_run
{
public:

typedef range<Char> range_type;
typedef std::vector<range_type> storage_type;

void swap(range_run& other);
bool test(Char v) const;
void set(range_type const& range);
void clear(range_type const& range);
void clear();

private:

storage_type run;
};
}}}}

#include <boost/spirit/home/support/char_set/range_run_impl.hpp>
#endif
