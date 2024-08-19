
#ifndef BOOST_XPRESSIVE_ALGORITHMS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_ALGORITHMS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <string>
#include <iterator>
#include <boost/mpl/or.hpp>
#include <boost/range/end.hpp>
#include <boost/range/begin.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/xpressive/match_results.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/utility/save_restore.hpp>

#define BOOST_XPR_NONDEDUCED_TYPE_(x) typename mpl::identity<x>::type

namespace boost { namespace xpressive
{


namespace detail
{
template<typename BidiIter>
inline bool regex_match_impl
(
BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<BidiIter> access;
BOOST_ASSERT(0 != re.regex_id());

detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
state.flags_.match_all_ = true;
state.sub_match(0).begin_ = begin;

if(access::match(re, state))
{
access::set_prefix_suffix(what, begin, end);
return true;
}

else if(state.found_partial_match_ && 0 != (flags & regex_constants::match_partial))
{
state.set_partial_match();
return true;
}

access::reset(what);
return false;
}
} 

template<typename BidiIter>
inline bool regex_match
(
BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename BidiIter>
inline bool regex_match
(
BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename Char>
inline bool regex_match
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *begin
, match_results<Char *> &what
, basic_regex<Char *> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<Char *> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

typedef typename remove_const<Char>::type char_type;
Char *end = begin + std::char_traits<char_type>::length(begin);
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_match
(
BidiRange &rng
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

BidiIter begin = boost::begin(rng), end = boost::end(rng);
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_match
(
BidiRange const &rng
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

BidiIter begin = boost::begin(rng), end = boost::end(rng);
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename Char>
inline bool regex_match
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *begin
, basic_regex<Char *> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
if(0 == re.regex_id())
{
return false;
}

match_results<Char *> what;
typedef typename remove_const<Char>::type char_type;
Char *end = begin + std::char_traits<char_type>::length(begin);
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_match
(
BidiRange &rng
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
BidiIter begin = boost::begin(rng), end = boost::end(rng);
return detail::regex_match_impl(begin, end, what, re, flags);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_match
(
BidiRange const &rng
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
BidiIter begin = boost::begin(rng), end = boost::end(rng);
return detail::regex_match_impl(begin, end, what, re, flags);
}



namespace detail
{
template<typename BidiIter>
inline bool regex_search_impl
(
match_state<BidiIter> &state
, basic_regex<BidiIter> const &re
, bool not_initial_null = false
)
{
typedef core_access<BidiIter> access;
match_results<BidiIter> &what = *state.context_.results_ptr_;
BOOST_ASSERT(0 != re.regex_id());

bool const partial_ok = state.flags_.match_partial_;
save_restore<bool> not_null(state.flags_.match_not_null_, state.flags_.match_not_null_ || not_initial_null);
state.flags_.match_prev_avail_ = state.flags_.match_prev_avail_ || !state.bos();

regex_impl<BidiIter> const &impl = *access::get_regex_impl(re);
BidiIter const begin = state.cur_, end = state.end_;
BidiIter &sub0begin = state.sub_match(0).begin_;
sub0begin = state.cur_;

if(state.flags_.match_continuous_)
{
if(access::match(re, state))
{
access::set_prefix_suffix(what, begin, end);
return true;
}

else if(partial_ok && state.found_partial_match_)
{
state.set_partial_match();
return true;
}
}

else if(impl.finder_ && (!partial_ok || impl.finder_->ok_for_partial_matches()))
{
finder<BidiIter> const &find = *impl.finder_;
if(find(state))
{
if(state.cur_ != begin)
{
not_null.restore();
}

do
{
sub0begin = state.cur_;
if(access::match(re, state))
{
access::set_prefix_suffix(what, begin, end);
return true;
}

else if(partial_ok && state.found_partial_match_)
{
state.set_partial_match();
return true;
}

BOOST_ASSERT(state.cur_ == sub0begin);
not_null.restore();
}
while(state.cur_ != state.end_ && (++state.cur_, find(state)));
}
}

else
{
for(;;)
{
if(access::match(re, state))
{
access::set_prefix_suffix(what, begin, end);
return true;
}

else if(partial_ok && state.found_partial_match_)
{
state.set_partial_match();
return true;
}

else if(end == sub0begin)
{
break;
}

BOOST_ASSERT(state.cur_ == sub0begin);
state.cur_ = ++sub0begin;
not_null.restore();
}
}

access::reset(what);
return false;
}
} 


template<typename BidiIter>
inline bool regex_search
(
BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename BidiIter>
inline bool regex_search
(
BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename Char>
inline bool regex_search
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *begin
, match_results<Char *> &what
, basic_regex<Char *> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<Char *> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

typedef typename remove_const<Char>::type char_type;
Char *end = begin + std::char_traits<char_type>::length(begin);
detail::match_state<Char *> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_search
(
BidiRange &rng
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

BidiIter begin = boost::begin(rng), end = boost::end(rng);
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_search
(
BidiRange const &rng
, match_results<BidiIter> &what
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
access::reset(what);
return false;
}

BidiIter begin = boost::begin(rng), end = boost::end(rng);
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename Char>
inline bool regex_search
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *begin
, basic_regex<Char *> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef detail::core_access<Char *> access;

if(0 == re.regex_id())
{
return false;
}

match_results<Char *> what;
typedef typename remove_const<Char>::type char_type;
Char *end = begin + std::char_traits<char_type>::length(begin);
detail::match_state<Char *> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_search
(
BidiRange &rng
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
BidiIter begin = boost::begin(rng), end = boost::end(rng);
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}

template<typename BidiRange, typename BidiIter>
inline bool regex_search
(
BidiRange const &rng
, basic_regex<BidiIter> const &re
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiRange> >::type * = 0
)
{
typedef detail::core_access<BidiIter> access;

if(0 == re.regex_id())
{
return false;
}

match_results<BidiIter> what;
BidiIter begin = boost::begin(rng), end = boost::end(rng);
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
return detail::regex_search_impl(state, re);
}



namespace detail
{
template<typename OutIter, typename BidiIter, typename Formatter>
inline OutIter regex_replace_impl
(
OutIter out
, BidiIter begin
, BidiIter end
, basic_regex<BidiIter> const &re
, Formatter const &format
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
using namespace regex_constants;
typedef detail::core_access<BidiIter> access;
BOOST_ASSERT(0 != re.regex_id());

BidiIter cur = begin;
match_results<BidiIter> what;
detail::match_state<BidiIter> state(begin, end, what, *access::get_regex_impl(re), flags);
bool const yes_copy = (0 == (flags & format_no_copy));

if(detail::regex_search_impl(state, re))
{
if(yes_copy)
{
out = std::copy(cur, what[0].first, out);
}

out = what.format(out, format, flags);
cur = state.cur_ = state.next_search_ = what[0].second;

if(0 == (flags & format_first_only))
{
bool not_null = (0 == what.length());
state.reset(what, *access::get_regex_impl(re));
while(detail::regex_search_impl(state, re, not_null))
{
if(yes_copy)
{
out = std::copy(cur, what[0].first, out);
}

access::set_prefix_suffix(what, begin, end);
out = what.format(out, format, flags);
cur = state.cur_ = state.next_search_ = what[0].second;
not_null = (0 == what.length());
state.reset(what, *access::get_regex_impl(re));
}
}
}

if(yes_copy)
{
out = std::copy(cur, end, out);
}

return out;
}
} 

template<typename OutIter, typename BidiIter, typename Formatter>
inline OutIter regex_replace
(
OutIter out
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, basic_regex<BidiIter> const &re
, Formatter const &format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<Formatter> >::type * = 0
)
{
if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
out = std::copy(begin, end, out);
}

return out;
}

return detail::regex_replace_impl(out, begin, end, re, format, flags);
}

template<typename OutIter, typename BidiIter>
inline OutIter regex_replace
(
OutIter out
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) begin
, BOOST_XPR_NONDEDUCED_TYPE_(BidiIter) end
, basic_regex<BidiIter> const &re
, typename iterator_value<BidiIter>::type const *format
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
out = std::copy(begin, end, out);
}

return out;
}

return detail::regex_replace_impl(out, begin, end, re, format, flags);
}

template<typename BidiContainer, typename BidiIter, typename Formatter>
inline BidiContainer regex_replace
(
BidiContainer &str
, basic_regex<BidiIter> const &re
, Formatter const &format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<mpl::or_<detail::is_char_ptr<BidiContainer>, detail::is_char_ptr<Formatter> > >::type * = 0
)
{
BidiContainer result;
BidiIter begin = boost::begin(str), end = boost::end(str);

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
std::copy(begin, end, std::back_inserter(result));
}

return result;
}

detail::regex_replace_impl(std::back_inserter(result), begin, end, re, format, flags);
return result;
}

template<typename BidiContainer, typename BidiIter, typename Formatter>
inline BidiContainer regex_replace
(
BidiContainer const &str
, basic_regex<BidiIter> const &re
, Formatter const &format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<mpl::or_<detail::is_char_ptr<BidiContainer>, detail::is_char_ptr<Formatter> > >::type * = 0
)
{
BidiContainer result;
BidiIter begin = boost::begin(str), end = boost::end(str);

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
std::copy(begin, end, std::back_inserter(result));
}

return result;
}

detail::regex_replace_impl(std::back_inserter(result), begin, end, re, format, flags);
return result;
}

template<typename Char, typename Formatter>
inline std::basic_string<typename remove_const<Char>::type> regex_replace
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *str
, basic_regex<Char *> const &re
, Formatter const &format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<Formatter> >::type * = 0
)
{
typedef typename remove_const<Char>::type char_type;
std::basic_string<char_type> result;

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
result = str;
}

return result;
}

Char *end = str + std::char_traits<char_type>::length(str);
detail::regex_replace_impl(std::back_inserter(result), str, end, re, format, flags);
return result;
}

template<typename BidiContainer, typename BidiIter>
inline BidiContainer regex_replace
(
BidiContainer &str
, basic_regex<BidiIter> const &re
, typename iterator_value<BidiIter>::type const *format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiContainer> >::type * = 0
)
{
BidiContainer result;
BidiIter begin = boost::begin(str), end = boost::end(str);

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
std::copy(begin, end, std::back_inserter(result));
}

return result;
}

detail::regex_replace_impl(std::back_inserter(result), begin, end, re, format, flags);
return result;
}

template<typename BidiContainer, typename BidiIter>
inline BidiContainer regex_replace
(
BidiContainer const &str
, basic_regex<BidiIter> const &re
, typename iterator_value<BidiIter>::type const *format
, regex_constants::match_flag_type flags = regex_constants::match_default
, typename disable_if<detail::is_char_ptr<BidiContainer> >::type * = 0
)
{
BidiContainer result;
BidiIter begin = boost::begin(str), end = boost::end(str);

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
std::copy(begin, end, std::back_inserter(result));
}

return result;
}

detail::regex_replace_impl(std::back_inserter(result), begin, end, re, format, flags);
return result;
}

template<typename Char>
inline std::basic_string<typename remove_const<Char>::type> regex_replace
(
BOOST_XPR_NONDEDUCED_TYPE_(Char) *str
, basic_regex<Char *> const &re
, typename add_const<Char>::type *format
, regex_constants::match_flag_type flags = regex_constants::match_default
)
{
typedef typename remove_const<Char>::type char_type;
std::basic_string<char_type> result;

if(0 == re.regex_id())
{
if((0 == (flags & regex_constants::format_no_copy)))
{
result = str;
}

return result;
}

Char *end = str + std::char_traits<char_type>::length(str);
detail::regex_replace_impl(std::back_inserter(result), str, end, re, format, flags);
return result;
}

}} 

#endif
