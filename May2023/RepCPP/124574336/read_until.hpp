
#ifndef BOOST_ASIO_IMPL_READ_UNTIL_HPP
#define BOOST_ASIO_IMPL_READ_UNTIL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <boost/asio/associated_allocator.hpp>
#include <boost/asio/associated_executor.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/buffers_iterator.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_cont_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_tracking.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/limits.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

namespace detail
{
template <typename Iterator1, typename Iterator2>
std::pair<Iterator1, bool> partial_search(
Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
{
for (Iterator1 iter1 = first1; iter1 != last1; ++iter1)
{
Iterator1 test_iter1 = iter1;
Iterator2 test_iter2 = first2;
for (;; ++test_iter1, ++test_iter2)
{
if (test_iter2 == last2)
return std::make_pair(iter1, true);
if (test_iter1 == last1)
{
if (test_iter2 != first2)
return std::make_pair(iter1, false);
else
break;
}
if (*test_iter1 != *test_iter2)
break;
}
}
return std::make_pair(last1, false);
}
} 

#if !defined(BOOST_ASIO_NO_DYNAMIC_BUFFER_V1)

template <typename SyncReadStream, typename DynamicBuffer_v1>
inline std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers, char delim,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), delim, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v1>
std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
char delim, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
typename decay<DynamicBuffer_v1>::type b(
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers));

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v1::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = b.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

iterator iter = std::find(start_pos, end, delim);
if (iter != end)
{
ec = boost::system::error_code();
return iter - begin + 1;
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
b.commit(s.read_some(b.prepare(bytes_to_read), ec));
if (ec)
return 0;
}
}

template <typename SyncReadStream, typename DynamicBuffer_v1>
inline std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
BOOST_ASIO_STRING_VIEW_PARAM delim,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), delim, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v1>
std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
BOOST_ASIO_STRING_VIEW_PARAM delim, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
typename decay<DynamicBuffer_v1>::type b(
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers));

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v1::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = b.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = detail::partial_search(
start_pos, end, delim.begin(), delim.end());
if (result.first != end)
{
if (result.second)
{
ec = boost::system::error_code();
return result.first - begin + delim.length();
}
else
{
search_position = result.first - begin;
}
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
b.commit(s.read_some(b.prepare(bytes_to_read), ec));
if (ec)
return 0;
}
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename DynamicBuffer_v1>
inline std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
const boost::regex& expr,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), expr, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v1>
std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
const boost::regex& expr, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
typename decay<DynamicBuffer_v1>::type b(
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers));

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v1::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = b.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

boost::match_results<iterator,
typename std::vector<boost::sub_match<iterator> >::allocator_type>
match_results;
if (regex_search(start_pos, end, match_results, expr,
boost::match_default | boost::match_partial))
{
if (match_results[0].matched)
{
ec = boost::system::error_code();
return match_results[0].second - begin;
}
else
{
search_position = match_results[0].first - begin;
}
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
b.commit(s.read_some(b.prepare(bytes_to_read), ec));
if (ec)
return 0;
}
}

#endif 

template <typename SyncReadStream,
typename DynamicBuffer_v1, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
MatchCondition match_condition,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
match_condition, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream,
typename DynamicBuffer_v1, typename MatchCondition>
std::size_t read_until(SyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
MatchCondition match_condition, boost::system::error_code& ec,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
typename decay<DynamicBuffer_v1>::type b(
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers));

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v1::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = b.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = match_condition(start_pos, end);
if (result.second)
{
ec = boost::system::error_code();
return result.first - begin;
}
else if (result.first != end)
{
search_position = result.first - begin;
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
b.commit(s.read_some(b.prepare(bytes_to_read), ec));
if (ec)
return 0;
}
}

#if !defined(BOOST_ASIO_NO_IOSTREAM)

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, char delim)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), delim);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, char delim,
boost::system::error_code& ec)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), delim, ec);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
BOOST_ASIO_STRING_VIEW_PARAM delim)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), delim);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
BOOST_ASIO_STRING_VIEW_PARAM delim, boost::system::error_code& ec)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), delim, ec);
}

#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, const boost::regex& expr)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), expr);
}

template <typename SyncReadStream, typename Allocator>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
boost::system::error_code& ec)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), expr, ec);
}

#endif 

template <typename SyncReadStream, typename Allocator, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, MatchCondition match_condition,
typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), match_condition);
}

template <typename SyncReadStream, typename Allocator, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
MatchCondition match_condition, boost::system::error_code& ec,
typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
return read_until(s, basic_streambuf_ref<Allocator>(b), match_condition, ec);
}

#endif 
#endif 
#endif 

template <typename SyncReadStream, typename DynamicBuffer_v2>
inline std::size_t read_until(SyncReadStream& s,
DynamicBuffer_v2 buffers, char delim,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), delim, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v2>
std::size_t read_until(SyncReadStream& s, DynamicBuffer_v2 buffers,
char delim, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
DynamicBuffer_v2& b = buffers;

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v2::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(b).data(0, b.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

iterator iter = std::find(start_pos, end, delim);
if (iter != end)
{
ec = boost::system::error_code();
return iter - begin + 1;
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
std::size_t pos = b.size();
b.grow(bytes_to_read);
std::size_t bytes_transferred = s.read_some(b.data(pos, bytes_to_read), ec);
b.shrink(bytes_to_read - bytes_transferred);
if (ec)
return 0;
}
}

template <typename SyncReadStream, typename DynamicBuffer_v2>
inline std::size_t read_until(SyncReadStream& s,
DynamicBuffer_v2 buffers, BOOST_ASIO_STRING_VIEW_PARAM delim,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), delim, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v2>
std::size_t read_until(SyncReadStream& s, DynamicBuffer_v2 buffers,
BOOST_ASIO_STRING_VIEW_PARAM delim, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
DynamicBuffer_v2& b = buffers;

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v2::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(b).data(0, b.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = detail::partial_search(
start_pos, end, delim.begin(), delim.end());
if (result.first != end)
{
if (result.second)
{
ec = boost::system::error_code();
return result.first - begin + delim.length();
}
else
{
search_position = result.first - begin;
}
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
std::size_t pos = b.size();
b.grow(bytes_to_read);
std::size_t bytes_transferred = s.read_some(b.data(pos, bytes_to_read), ec);
b.shrink(bytes_to_read - bytes_transferred);
if (ec)
return 0;
}
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

template <typename SyncReadStream, typename DynamicBuffer_v2>
inline std::size_t read_until(SyncReadStream& s,
DynamicBuffer_v2 buffers, const boost::regex& expr,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), expr, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream, typename DynamicBuffer_v2>
std::size_t read_until(SyncReadStream& s, DynamicBuffer_v2 buffers,
const boost::regex& expr, boost::system::error_code& ec,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
DynamicBuffer_v2& b = buffers;

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v2::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(b).data(0, b.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

boost::match_results<iterator,
typename std::vector<boost::sub_match<iterator> >::allocator_type>
match_results;
if (regex_search(start_pos, end, match_results, expr,
boost::match_default | boost::match_partial))
{
if (match_results[0].matched)
{
ec = boost::system::error_code();
return match_results[0].second - begin;
}
else
{
search_position = match_results[0].first - begin;
}
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
std::size_t pos = b.size();
b.grow(bytes_to_read);
std::size_t bytes_transferred = s.read_some(b.data(pos, bytes_to_read), ec);
b.shrink(bytes_to_read - bytes_transferred);
if (ec)
return 0;
}
}

#endif 

template <typename SyncReadStream,
typename DynamicBuffer_v2, typename MatchCondition>
inline std::size_t read_until(SyncReadStream& s,
DynamicBuffer_v2 buffers, MatchCondition match_condition,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
boost::system::error_code ec;
std::size_t bytes_transferred = read_until(s,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
match_condition, ec);
boost::asio::detail::throw_error(ec, "read_until");
return bytes_transferred;
}

template <typename SyncReadStream,
typename DynamicBuffer_v2, typename MatchCondition>
std::size_t read_until(SyncReadStream& s, DynamicBuffer_v2 buffers,
MatchCondition match_condition, boost::system::error_code& ec,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
DynamicBuffer_v2& b = buffers;

std::size_t search_position = 0;
for (;;)
{
typedef typename DynamicBuffer_v2::const_buffers_type buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(b).data(0, b.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = match_condition(start_pos, end);
if (result.second)
{
ec = boost::system::error_code();
return result.first - begin;
}
else if (result.first != end)
{
search_position = result.first - begin;
}
else
{
search_position = end - begin;
}

if (b.size() == b.max_size())
{
ec = error::not_found;
return 0;
}

std::size_t bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512, b.capacity() - b.size()),
std::min<std::size_t>(65536, b.max_size() - b.size()));
std::size_t pos = b.size();
b.grow(bytes_to_read);
std::size_t bytes_transferred = s.read_some(b.data(pos, bytes_to_read), ec);
b.shrink(bytes_to_read - bytes_transferred);
if (ec)
return 0;
}
}

#endif 

#if !defined(BOOST_ASIO_NO_DYNAMIC_BUFFER_V1)

namespace detail
{
template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
class read_until_delim_op_v1
{
public:
template <typename BufferSequence>
read_until_delim_op_v1(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
char delim, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
delim_(delim),
start_(0),
search_position_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_delim_op_v1(const read_until_delim_op_v1& other)
: stream_(other.stream_),
buffers_(other.buffers_),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
handler_(other.handler_)
{
}

read_until_delim_op_v1(read_until_delim_op_v1&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(other.buffers_)),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t bytes_to_read;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v1::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = buffers_.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

iterator iter = std::find(start_pos, end, delim_);
if (iter != end)
{
search_position_ = iter - begin + 1;
bytes_to_read = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read = 0;
}

else
{
search_position_ = end - begin;
bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read == 0)
break;

{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.prepare(bytes_to_read),
BOOST_ASIO_MOVE_CAST(read_until_delim_op_v1)(*this));
}
return; default:
buffers_.commit(bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v1 buffers_;
char delim_;
int start_;
std::size_t search_position_;
ReadHandler handler_;
};

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_delim_v1
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_delim_v1(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v1>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
char delim) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_delim_op_v1<AsyncReadStream,
typename decay<DynamicBuffer_v1>::type,
typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
delim, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_delim_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream, typename DynamicBuffer_v1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
char delim, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_delim_v1<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), delim);
}

namespace detail
{
template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
class read_until_delim_string_op_v1
{
public:
template <typename BufferSequence>
read_until_delim_string_op_v1(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
const std::string& delim, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
delim_(delim),
start_(0),
search_position_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_delim_string_op_v1(const read_until_delim_string_op_v1& other)
: stream_(other.stream_),
buffers_(other.buffers_),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
handler_(other.handler_)
{
}

read_until_delim_string_op_v1(read_until_delim_string_op_v1&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(other.buffers_)),
delim_(BOOST_ASIO_MOVE_CAST(std::string)(other.delim_)),
start_(other.start_),
search_position_(other.search_position_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t bytes_to_read;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v1::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = buffers_.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = detail::partial_search(
start_pos, end, delim_.begin(), delim_.end());
if (result.first != end && result.second)
{
search_position_ = result.first - begin + delim_.length();
bytes_to_read = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read = 0;
}

else
{
if (result.first != end)
{
search_position_ = result.first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read == 0)
break;

{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.prepare(bytes_to_read),
BOOST_ASIO_MOVE_CAST(read_until_delim_string_op_v1)(*this));
}
return; default:
buffers_.commit(bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v1 buffers_;
std::string delim_;
int start_;
std::size_t search_position_;
ReadHandler handler_;
};

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_delim_string_v1
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_delim_string_v1(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v1>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
const std::string& delim) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_delim_string_op_v1<AsyncReadStream,
typename decay<DynamicBuffer_v1>::type,
typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
delim, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_delim_string_op_v1<AsyncReadStream,
DynamicBuffer_v1, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream, typename DynamicBuffer_v1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
BOOST_ASIO_STRING_VIEW_PARAM delim,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_delim_string_v1<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
static_cast<std::string>(delim));
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

namespace detail
{
template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler>
class read_until_expr_op_v1
{
public:
template <typename BufferSequence>
read_until_expr_op_v1(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
const boost::regex& expr, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
expr_(expr),
start_(0),
search_position_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_expr_op_v1(const read_until_expr_op_v1& other)
: stream_(other.stream_),
buffers_(other.buffers_),
expr_(other.expr_),
start_(other.start_),
search_position_(other.search_position_),
handler_(other.handler_)
{
}

read_until_expr_op_v1(read_until_expr_op_v1&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(other.buffers_)),
expr_(other.expr_),
start_(other.start_),
search_position_(other.search_position_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t bytes_to_read;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v1::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = buffers_.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

boost::match_results<iterator,
typename std::vector<boost::sub_match<iterator> >::allocator_type>
match_results;
bool match = regex_search(start_pos, end, match_results, expr_,
boost::match_default | boost::match_partial);
if (match && match_results[0].matched)
{
search_position_ = match_results[0].second - begin;
bytes_to_read = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read = 0;
}

else
{
if (match)
{
search_position_ = match_results[0].first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read == 0)
break;

{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.prepare(bytes_to_read),
BOOST_ASIO_MOVE_CAST(read_until_expr_op_v1)(*this));
}
return; default:
buffers_.commit(bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v1 buffers_;
RegEx expr_;
int start_;
std::size_t search_position_;
ReadHandler handler_;
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename RegEx, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename RegEx, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_expr_v1
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_expr_v1(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v1, typename RegEx>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers, const RegEx& expr) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_expr_op_v1<AsyncReadStream,
typename decay<DynamicBuffer_v1>::type,
RegEx, typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
expr, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename RegEx, typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_expr_op_v1<AsyncReadStream,
DynamicBuffer_v1, RegEx, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream, typename DynamicBuffer_v1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
const boost::regex& expr,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_expr_v1<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), expr);
}

#endif 

namespace detail
{
template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler>
class read_until_match_op_v1
{
public:
template <typename BufferSequence>
read_until_match_op_v1(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
MatchCondition match_condition, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
match_condition_(match_condition),
start_(0),
search_position_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_match_op_v1(const read_until_match_op_v1& other)
: stream_(other.stream_),
buffers_(other.buffers_),
match_condition_(other.match_condition_),
start_(other.start_),
search_position_(other.search_position_),
handler_(other.handler_)
{
}

read_until_match_op_v1(read_until_match_op_v1&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(other.buffers_)),
match_condition_(other.match_condition_),
start_(other.start_),
search_position_(other.search_position_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t bytes_to_read;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v1::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers = buffers_.data();
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = match_condition_(start_pos, end);
if (result.second)
{
search_position_ = result.first - begin;
bytes_to_read = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read = 0;
}

else
{
if (result.first != end)
{
search_position_ = result.first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read == 0)
break;

{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.prepare(bytes_to_read),
BOOST_ASIO_MOVE_CAST(read_until_match_op_v1)(*this));
}
return; default:
buffers_.commit(bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v1 buffers_;
MatchCondition match_condition_;
int start_;
std::size_t search_position_;
ReadHandler handler_;
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_match_op_v1<AsyncReadStream, DynamicBuffer_v1,
MatchCondition, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_match_op_v1<AsyncReadStream, DynamicBuffer_v1,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_match_op_v1<AsyncReadStream, DynamicBuffer_v1,
MatchCondition, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename MatchCondition,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_match_op_v1<AsyncReadStream, DynamicBuffer_v1,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v1, typename MatchCondition,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_match_op_v1<AsyncReadStream, DynamicBuffer_v1,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_match_v1
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_match_v1(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler,
typename DynamicBuffer_v1, typename MatchCondition>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
MatchCondition match_condition) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_match_op_v1<AsyncReadStream,
typename decay<DynamicBuffer_v1>::type,
MatchCondition, typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers),
match_condition, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_match_op_v1<AsyncReadStream,
DynamicBuffer_v1, MatchCondition, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_match_op_v1<AsyncReadStream,
DynamicBuffer_v1, MatchCondition, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v1,
typename MatchCondition, typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_match_op_v1<AsyncReadStream,
DynamicBuffer_v1, MatchCondition, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_match_op_v1<AsyncReadStream,
DynamicBuffer_v1, MatchCondition, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream,
typename DynamicBuffer_v1, typename MatchCondition,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v1) buffers,
MatchCondition match_condition, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v1<typename decay<DynamicBuffer_v1>::type>::value
&& !is_dynamic_buffer_v2<typename decay<DynamicBuffer_v1>::type>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_match_v1<AsyncReadStream>(s), handler,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v1)(buffers), match_condition);
}

#if !defined(BOOST_ASIO_NO_IOSTREAM)

template <typename AsyncReadStream, typename Allocator,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
char delim, BOOST_ASIO_MOVE_ARG(ReadHandler) handler)
{
return async_read_until(s, basic_streambuf_ref<Allocator>(b),
delim, BOOST_ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename AsyncReadStream, typename Allocator,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
BOOST_ASIO_STRING_VIEW_PARAM delim,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler)
{
return async_read_until(s, basic_streambuf_ref<Allocator>(b),
delim, BOOST_ASIO_MOVE_CAST(ReadHandler)(handler));
}

#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

template <typename AsyncReadStream, typename Allocator,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler)
{
return async_read_until(s, basic_streambuf_ref<Allocator>(b),
expr, BOOST_ASIO_MOVE_CAST(ReadHandler)(handler));
}

#endif 

template <typename AsyncReadStream, typename Allocator, typename MatchCondition,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
boost::asio::basic_streambuf<Allocator>& b,
MatchCondition match_condition, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<is_match_condition<MatchCondition>::value>::type*)
{
return async_read_until(s, basic_streambuf_ref<Allocator>(b),
match_condition, BOOST_ASIO_MOVE_CAST(ReadHandler)(handler));
}

#endif 
#endif 
#endif 

namespace detail
{
template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
class read_until_delim_op_v2
{
public:
template <typename BufferSequence>
read_until_delim_op_v2(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
char delim, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
delim_(delim),
start_(0),
search_position_(0),
bytes_to_read_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_delim_op_v2(const read_until_delim_op_v2& other)
: stream_(other.stream_),
buffers_(other.buffers_),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(other.handler_)
{
}

read_until_delim_op_v2(read_until_delim_op_v2&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(other.buffers_)),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t pos;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v2::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(buffers_).data(
0, buffers_.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

iterator iter = std::find(start_pos, end, delim_);
if (iter != end)
{
search_position_ = iter - begin + 1;
bytes_to_read_ = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read_ = 0;
}

else
{
search_position_ = end - begin;
bytes_to_read_ = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read_ == 0)
break;

pos = buffers_.size();
buffers_.grow(bytes_to_read_);
{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.data(pos, bytes_to_read_),
BOOST_ASIO_MOVE_CAST(read_until_delim_op_v2)(*this));
}
return; default:
buffers_.shrink(bytes_to_read_ - bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v2 buffers_;
char delim_;
int start_;
std::size_t search_position_;
std::size_t bytes_to_read_;
ReadHandler handler_;
};

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_delim_v2
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_delim_v2(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v2>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v2) buffers, char delim) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_delim_op_v2<AsyncReadStream,
typename decay<DynamicBuffer_v2>::type,
typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
delim, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_delim_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream, typename DynamicBuffer_v2,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s, DynamicBuffer_v2 buffers,
char delim, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_delim_v2<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), delim);
}

namespace detail
{
template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
class read_until_delim_string_op_v2
{
public:
template <typename BufferSequence>
read_until_delim_string_op_v2(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
const std::string& delim, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
delim_(delim),
start_(0),
search_position_(0),
bytes_to_read_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_delim_string_op_v2(const read_until_delim_string_op_v2& other)
: stream_(other.stream_),
buffers_(other.buffers_),
delim_(other.delim_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(other.handler_)
{
}

read_until_delim_string_op_v2(read_until_delim_string_op_v2&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(other.buffers_)),
delim_(BOOST_ASIO_MOVE_CAST(std::string)(other.delim_)),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t pos;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v2::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(buffers_).data(
0, buffers_.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = detail::partial_search(
start_pos, end, delim_.begin(), delim_.end());
if (result.first != end && result.second)
{
search_position_ = result.first - begin + delim_.length();
bytes_to_read_ = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read_ = 0;
}

else
{
if (result.first != end)
{
search_position_ = result.first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read_ = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read_ == 0)
break;

pos = buffers_.size();
buffers_.grow(bytes_to_read_);
{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.data(pos, bytes_to_read_),
BOOST_ASIO_MOVE_CAST(read_until_delim_string_op_v2)(*this));
}
return; default:
buffers_.shrink(bytes_to_read_ - bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v2 buffers_;
std::string delim_;
int start_;
std::size_t search_position_;
std::size_t bytes_to_read_;
ReadHandler handler_;
};

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_delim_string_v2
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_delim_string_v2(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v2>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v2) buffers,
const std::string& delim) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_delim_string_op_v2<AsyncReadStream,
typename decay<DynamicBuffer_v2>::type,
typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
delim, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_delim_string_op_v2<AsyncReadStream,
DynamicBuffer_v2, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream,
typename DynamicBuffer_v2,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s,
DynamicBuffer_v2 buffers, BOOST_ASIO_STRING_VIEW_PARAM delim,
BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_delim_string_v2<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
static_cast<std::string>(delim));
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

namespace detail
{
template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler>
class read_until_expr_op_v2
{
public:
template <typename BufferSequence>
read_until_expr_op_v2(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
const boost::regex& expr, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
expr_(expr),
start_(0),
search_position_(0),
bytes_to_read_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_expr_op_v2(const read_until_expr_op_v2& other)
: stream_(other.stream_),
buffers_(other.buffers_),
expr_(other.expr_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(other.handler_)
{
}

read_until_expr_op_v2(read_until_expr_op_v2&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(other.buffers_)),
expr_(other.expr_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t pos;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v2::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(buffers_).data(
0, buffers_.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

boost::match_results<iterator,
typename std::vector<boost::sub_match<iterator> >::allocator_type>
match_results;
bool match = regex_search(start_pos, end, match_results, expr_,
boost::match_default | boost::match_partial);
if (match && match_results[0].matched)
{
search_position_ = match_results[0].second - begin;
bytes_to_read_ = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read_ = 0;
}

else
{
if (match)
{
search_position_ = match_results[0].first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read_ = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read_ == 0)
break;

pos = buffers_.size();
buffers_.grow(bytes_to_read_);
{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.data(pos, bytes_to_read_),
BOOST_ASIO_MOVE_CAST(read_until_expr_op_v2)(*this));
}
return; default:
buffers_.shrink(bytes_to_read_ - bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v2 buffers_;
RegEx expr_;
int start_;
std::size_t search_position_;
std::size_t bytes_to_read_;
ReadHandler handler_;
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename RegEx, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename RegEx, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_expr_v2
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_expr_v2(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler, typename DynamicBuffer_v2, typename RegEx>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v2) buffers,
const RegEx& expr) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_expr_op_v2<AsyncReadStream,
typename decay<DynamicBuffer_v2>::type,
RegEx, typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
expr, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename RegEx, typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_expr_op_v2<AsyncReadStream,
DynamicBuffer_v2, RegEx, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream, typename DynamicBuffer_v2,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s, DynamicBuffer_v2 buffers,
const boost::regex& expr, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_expr_v2<AsyncReadStream>(s),
handler, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), expr);
}

#endif 

namespace detail
{
template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler>
class read_until_match_op_v2
{
public:
template <typename BufferSequence>
read_until_match_op_v2(AsyncReadStream& stream,
BOOST_ASIO_MOVE_ARG(BufferSequence) buffers,
MatchCondition match_condition, ReadHandler& handler)
: stream_(stream),
buffers_(BOOST_ASIO_MOVE_CAST(BufferSequence)(buffers)),
match_condition_(match_condition),
start_(0),
search_position_(0),
bytes_to_read_(0),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
read_until_match_op_v2(const read_until_match_op_v2& other)
: stream_(other.stream_),
buffers_(other.buffers_),
match_condition_(other.match_condition_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(other.handler_)
{
}

read_until_match_op_v2(read_until_match_op_v2&& other)
: stream_(other.stream_),
buffers_(BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(other.buffers_)),
match_condition_(other.match_condition_),
start_(other.start_),
search_position_(other.search_position_),
bytes_to_read_(other.bytes_to_read_),
handler_(BOOST_ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
const std::size_t not_found = (std::numeric_limits<std::size_t>::max)();
std::size_t pos;
switch (start_ = start)
{
case 1:
for (;;)
{
{
typedef typename DynamicBuffer_v2::const_buffers_type
buffers_type;
typedef buffers_iterator<buffers_type> iterator;
buffers_type data_buffers =
const_cast<const DynamicBuffer_v2&>(buffers_).data(
0, buffers_.size());
iterator begin = iterator::begin(data_buffers);
iterator start_pos = begin + search_position_;
iterator end = iterator::end(data_buffers);

std::pair<iterator, bool> result = match_condition_(start_pos, end);
if (result.second)
{
search_position_ = result.first - begin;
bytes_to_read_ = 0;
}

else if (buffers_.size() == buffers_.max_size())
{
search_position_ = not_found;
bytes_to_read_ = 0;
}

else
{
if (result.first != end)
{
search_position_ = result.first - begin;
}
else
{
search_position_ = end - begin;
}

bytes_to_read_ = std::min<std::size_t>(
std::max<std::size_t>(512,
buffers_.capacity() - buffers_.size()),
std::min<std::size_t>(65536,
buffers_.max_size() - buffers_.size()));
}
}

if (!start && bytes_to_read_ == 0)
break;

pos = buffers_.size();
buffers_.grow(bytes_to_read_);
{
BOOST_ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, "async_read_until"));
stream_.async_read_some(buffers_.data(pos, bytes_to_read_),
BOOST_ASIO_MOVE_CAST(read_until_match_op_v2)(*this));
}
return; default:
buffers_.shrink(bytes_to_read_ - bytes_transferred);
if (ec || bytes_transferred == 0)
break;
}

const boost::system::error_code result_ec =
(search_position_ == not_found)
? error::not_found : ec;

const std::size_t result_n =
(ec || search_position_ == not_found)
? 0 : search_position_;

handler_(result_ec, result_n);
}
}

AsyncReadStream& stream_;
DynamicBuffer_v2 buffers_;
MatchCondition match_condition_;
int start_;
std::size_t search_position_;
std::size_t bytes_to_read_;
ReadHandler handler_;
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
read_until_match_op_v2<AsyncReadStream, DynamicBuffer_v2,
MatchCondition, ReadHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
read_until_match_op_v2<AsyncReadStream, DynamicBuffer_v2,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler>
inline bool asio_handler_is_continuation(
read_until_match_op_v2<AsyncReadStream, DynamicBuffer_v2,
MatchCondition, ReadHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename MatchCondition,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
read_until_match_op_v2<AsyncReadStream, DynamicBuffer_v2,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncReadStream,
typename DynamicBuffer_v2, typename MatchCondition,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
read_until_match_op_v2<AsyncReadStream, DynamicBuffer_v2,
MatchCondition, ReadHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncReadStream>
class initiate_async_read_until_match_v2
{
public:
typedef typename AsyncReadStream::executor_type executor_type;

explicit initiate_async_read_until_match_v2(AsyncReadStream& stream)
: stream_(stream)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return stream_.get_executor();
}

template <typename ReadHandler,
typename DynamicBuffer_v2, typename MatchCondition>
void operator()(BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
BOOST_ASIO_MOVE_ARG(DynamicBuffer_v2) buffers,
MatchCondition match_condition) const
{
BOOST_ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
read_until_match_op_v2<AsyncReadStream,
typename decay<DynamicBuffer_v2>::type,
MatchCondition, typename decay<ReadHandler>::type>(
stream_, BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers),
match_condition, handler2.value)(boost::system::error_code(), 0, 1);
}

private:
AsyncReadStream& stream_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler, typename Allocator>
struct associated_allocator<
detail::read_until_match_op_v2<AsyncReadStream,
DynamicBuffer_v2, MatchCondition, ReadHandler>,
Allocator>
{
typedef typename associated_allocator<ReadHandler, Allocator>::type type;

static type get(
const detail::read_until_match_op_v2<AsyncReadStream,
DynamicBuffer_v2, MatchCondition, ReadHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<ReadHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncReadStream, typename DynamicBuffer_v2,
typename MatchCondition, typename ReadHandler, typename Executor>
struct associated_executor<
detail::read_until_match_op_v2<AsyncReadStream,
DynamicBuffer_v2, MatchCondition, ReadHandler>,
Executor>
: detail::associated_executor_forwarding_base<ReadHandler, Executor>
{
typedef typename associated_executor<ReadHandler, Executor>::type type;

static type get(
const detail::read_until_match_op_v2<AsyncReadStream,
DynamicBuffer_v2, MatchCondition, ReadHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<ReadHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncReadStream,
typename DynamicBuffer_v2, typename MatchCondition,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) ReadHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (boost::system::error_code, std::size_t))
async_read_until(AsyncReadStream& s, DynamicBuffer_v2 buffers,
MatchCondition match_condition, BOOST_ASIO_MOVE_ARG(ReadHandler) handler,
typename enable_if<
is_match_condition<MatchCondition>::value
&& is_dynamic_buffer_v2<DynamicBuffer_v2>::value
>::type*)
{
return async_initiate<ReadHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_read_until_match_v2<AsyncReadStream>(s), handler,
BOOST_ASIO_MOVE_CAST(DynamicBuffer_v2)(buffers), match_condition);
}

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
