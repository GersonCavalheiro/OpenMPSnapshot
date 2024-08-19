
#if !defined(BOOST_SPIRIT_RANGE_RUN_MAY_16_2006_0807_PM)
#define BOOST_SPIRIT_RANGE_RUN_MAY_16_2006_0807_PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/char_set/range_functions.hpp>
#include <boost/assert.hpp>
#include <algorithm>

namespace boost { namespace spirit { namespace support { namespace detail
{
template <typename Run, typename Iterator, typename Range>
inline bool
try_merge(Run& run, Iterator iter, Range const& range)
{
if (can_merge(*iter, range))
{
merge(*iter, range);

Iterator i = iter+1;
while (i != run.end() && i->last <= iter->last)
++i;
if (i != run.end() && i->first-1 <= iter->last)
{
iter->last = i->last;
++i;
}

run.erase(iter+1, i);
return true;
}
return false;
}

template <typename Char>
inline bool
range_run<Char>::test(Char val) const
{
if (run.empty())
return false;

typename storage_type::const_iterator iter =
std::upper_bound(
run.begin(), run.end(), val,
range_compare<range_type>()
);

return iter != run.begin() && includes(*(--iter), val);
}

template <typename Char>
inline void
range_run<Char>::swap(range_run& other)
{
run.swap(other.run);
}

template <typename Char>
void
range_run<Char>::set(range_type const& range)
{
BOOST_ASSERT(is_valid(range));
if (run.empty())
{
run.push_back(range);
return;
}

typename storage_type::iterator iter =
std::upper_bound(
run.begin(), run.end(), range,
range_compare<range_type>()
);

if (iter != run.begin())
{
if (includes(*(iter-1), range))
{
return;
}

if (try_merge(run, iter-1, range))
{
return;
}
}

if (iter == run.end() || !try_merge(run, iter, range))
{
run.insert(iter, range);
}
}

template <typename Char>
void
range_run<Char>::clear(range_type const& range)
{
BOOST_ASSERT(is_valid(range));
if (!run.empty())
{
typename storage_type::iterator iter =
std::upper_bound(
run.begin(), run.end(), range,
range_compare<range_type>()
);

if (iter != run.begin())
{
typename storage_type::iterator const left_iter = iter-1;

if (left_iter->first < range.first)
{
if (left_iter->last > range.last)
{
Char save_last = left_iter->last;
left_iter->last = range.first-1;
run.insert(iter, range_type(range.last+1, save_last));
return;
}
else if (left_iter->last >= range.first)
{
left_iter->last = range.first-1;
}
}

else
{
iter = left_iter;
}
}

typename storage_type::iterator i = iter;
while (i != run.end() && i->last <= range.last)
++i;
if (i != run.end() && i->first <= range.last)
i->first = range.last+1;

run.erase(iter, i);
}
}

template <typename Char>
inline void
range_run<Char>::clear()
{
run.clear();
}
}}}}

#endif
