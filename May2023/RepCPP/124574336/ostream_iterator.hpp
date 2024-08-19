
#if !defined(BOOST_SPIRIT_KARMA_OSTREAM_ITERATOR_MAY_26_2007_1016PM)
#define BOOST_SPIRIT_KARMA_OSTREAM_ITERATOR_MAY_26_2007_1016PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <iterator>

namespace boost { namespace spirit { namespace karma 
{
template <
typename T, typename Elem = char
, typename Traits = std::char_traits<Elem> >
class ostream_iterator 
{
public:
typedef std::output_iterator_tag iterator_category;
typedef void value_type;
typedef void difference_type;
typedef void pointer;
typedef void reference;
typedef Elem char_type;
typedef Traits traits_type;
typedef std::basic_ostream<Elem, Traits> ostream_type;
typedef ostream_iterator<T, Elem, Traits> self_type;

ostream_iterator(ostream_type& os_, Elem const* delim_ = 0)
: os(&os_), delim(delim_) {}

self_type& operator= (T const& val)
{
*os << val;
if (0 != delim)
*os << delim;
return *this;
}

self_type& operator*() { return *this; }
self_type& operator++() { return *this; }
self_type operator++(int) { return *this; }

ostream_type& get_ostream() { return *os; }
ostream_type const& get_ostream() const { return *os; }

bool good() const { return get_ostream().good(); }

protected:
ostream_type *os;
Elem const* delim;
};

}}}

#endif
