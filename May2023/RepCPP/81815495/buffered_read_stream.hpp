
#ifndef ASIO_IMPL_BUFFERED_READ_STREAM_HPP
#define ASIO_IMPL_BUFFERED_READ_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/associator.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Stream>
std::size_t buffered_read_stream<Stream>::fill()
{
detail::buffer_resize_guard<detail::buffered_stream_storage>
resize_guard(storage_);
std::size_t previous_size = storage_.size();
storage_.resize(storage_.capacity());
storage_.resize(previous_size + next_layer_.read_some(buffer(
storage_.data() + previous_size,
storage_.size() - previous_size)));
resize_guard.commit();
return storage_.size() - previous_size;
}

template <typename Stream>
std::size_t buffered_read_stream<Stream>::fill(asio::error_code& ec)
{
detail::buffer_resize_guard<detail::buffered_stream_storage>
resize_guard(storage_);
std::size_t previous_size = storage_.size();
storage_.resize(storage_.capacity());
storage_.resize(previous_size + next_layer_.read_some(buffer(
storage_.data() + previous_size,
storage_.size() - previous_size),
ec));
resize_guard.commit();
return storage_.size() - previous_size;
}

namespace detail
{
template <typename ReadHandler>
class buffered_fill_handler
{
public:
buffered_fill_handler(detail::buffered_stream_storage& storage,
std::size_t previous_size, ReadHandler& handler)
: storage_(storage),
previous_size_(previous_size),
handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(ASIO_HAS_MOVE)
buffered_fill_handler(const buffered_fill_handler& other)
: storage_(other.storage_),
previous_size_(other.previous_size_),
handler_(other.handler_)
{
}

buffered_fill_handler(buffered_fill_handler&& other)
: storage_(other.storage_),
previous_size_(other.previous_size_),
handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const asio::error_code& ec,
const std::size_t bytes_transferred)
{
storage_.resize(previous_size_ + bytes_transferred);
ASIO_MOVE_OR_LVALUE(ReadHandler)(handler_)(ec, bytes_transferred);
}

detail::buffered_stream_storage& storage_;
std::size_t previous_size_;
ReadHandler handler_;
};

template <typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
buffered_fill_handler<ReadHandler>* this_handler)
{
#if defined(ASIO_NO_DEPRECATED)
asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
buffered_fill_handler<ReadHandler>* this_handler)
{
asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename ReadHandler>
inline bool asio_handler_is_continuation(
buffered_fill_handler<ReadHandler>* this_handler)
{
return asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
buffered_fill_handler<ReadHandler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
buffered_fill_handler<ReadHandler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Stream>
class initiate_async_buffered_fill
{
public:
typedef typename remove_reference<
Stream>::type::lowest_layer_type::executor_type executor_type;

explicit initiate_async_buffered_fill(
typename remove_reference<Stream>::type& next_layer)
: next_layer_(next_layer)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return next_layer_.lowest_layer().get_executor();
}

template <typename ReadHandler>
void operator()(ASIO_MOVE_ARG(ReadHandler) handler,
buffered_stream_storage* storage) const
{
ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

non_const_lvalue<ReadHandler> handler2(handler);
std::size_t previous_size = storage->size();
storage->resize(storage->capacity());
next_layer_.async_read_some(
buffer(
storage->data() + previous_size,
storage->size() - previous_size),
buffered_fill_handler<typename decay<ReadHandler>::type>(
*storage, previous_size, handler2.value));
}

private:
typename remove_reference<Stream>::type& next_layer_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <template <typename, typename> class Associator,
typename ReadHandler, typename DefaultCandidate>
struct associator<Associator,
detail::buffered_fill_handler<ReadHandler>,
DefaultCandidate>
: Associator<ReadHandler, DefaultCandidate>
{
static typename Associator<ReadHandler, DefaultCandidate>::type get(
const detail::buffered_fill_handler<ReadHandler>& h,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<ReadHandler, DefaultCandidate>::get(h.handler_, c);
}
};

#endif 

template <typename Stream>
template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
buffered_read_stream<Stream>::async_fill(
ASIO_MOVE_ARG(ReadHandler) handler)
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
detail::initiate_async_buffered_fill<Stream>(next_layer_),
handler, &storage_);
}

template <typename Stream>
template <typename MutableBufferSequence>
std::size_t buffered_read_stream<Stream>::read_some(
const MutableBufferSequence& buffers)
{
using asio::buffer_size;
if (buffer_size(buffers) == 0)
return 0;

if (storage_.empty())
this->fill();

return this->copy(buffers);
}

template <typename Stream>
template <typename MutableBufferSequence>
std::size_t buffered_read_stream<Stream>::read_some(
const MutableBufferSequence& buffers, asio::error_code& ec)
{
ec = asio::error_code();

using asio::buffer_size;
if (buffer_size(buffers) == 0)
return 0;

if (storage_.empty() && !this->fill(ec))
return 0;

return this->copy(buffers);
}

namespace detail
{
template <typename MutableBufferSequence, typename ReadHandler>
class buffered_read_some_handler
{
public:
buffered_read_some_handler(detail::buffered_stream_storage& storage,
const MutableBufferSequence& buffers, ReadHandler& handler)
: storage_(storage),
buffers_(buffers),
handler_(ASIO_MOVE_CAST(ReadHandler)(handler))
{
}

#if defined(ASIO_HAS_MOVE)
buffered_read_some_handler(const buffered_read_some_handler& other)
: storage_(other.storage_),
buffers_(other.buffers_),
handler_(other.handler_)
{
}

buffered_read_some_handler(buffered_read_some_handler&& other)
: storage_(other.storage_),
buffers_(other.buffers_),
handler_(ASIO_MOVE_CAST(ReadHandler)(other.handler_))
{
}
#endif 

void operator()(const asio::error_code& ec, std::size_t)
{
if (ec || storage_.empty())
{
const std::size_t length = 0;
ASIO_MOVE_OR_LVALUE(ReadHandler)(handler_)(ec, length);
}
else
{
const std::size_t bytes_copied = asio::buffer_copy(
buffers_, storage_.data(), storage_.size());
storage_.consume(bytes_copied);
ASIO_MOVE_OR_LVALUE(ReadHandler)(handler_)(ec, bytes_copied);
}
}

detail::buffered_stream_storage& storage_;
MutableBufferSequence buffers_;
ReadHandler handler_;
};

template <typename MutableBufferSequence, typename ReadHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
buffered_read_some_handler<
MutableBufferSequence, ReadHandler>* this_handler)
{
#if defined(ASIO_NO_DEPRECATED)
asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename MutableBufferSequence, typename ReadHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
buffered_read_some_handler<
MutableBufferSequence, ReadHandler>* this_handler)
{
asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename MutableBufferSequence, typename ReadHandler>
inline bool asio_handler_is_continuation(
buffered_read_some_handler<
MutableBufferSequence, ReadHandler>* this_handler)
{
return asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename MutableBufferSequence,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
buffered_read_some_handler<
MutableBufferSequence, ReadHandler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename MutableBufferSequence,
typename ReadHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
buffered_read_some_handler<
MutableBufferSequence, ReadHandler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Stream>
class initiate_async_buffered_read_some
{
public:
typedef typename remove_reference<
Stream>::type::lowest_layer_type::executor_type executor_type;

explicit initiate_async_buffered_read_some(
typename remove_reference<Stream>::type& next_layer)
: next_layer_(next_layer)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return next_layer_.lowest_layer().get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(ASIO_MOVE_ARG(ReadHandler) handler,
buffered_stream_storage* storage,
const MutableBufferSequence& buffers) const
{
ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

using asio::buffer_size;
non_const_lvalue<ReadHandler> handler2(handler);
if (buffer_size(buffers) == 0 || !storage->empty())
{
next_layer_.async_read_some(ASIO_MUTABLE_BUFFER(0, 0),
buffered_read_some_handler<MutableBufferSequence,
typename decay<ReadHandler>::type>(
*storage, buffers, handler2.value));
}
else
{
initiate_async_buffered_fill<Stream>(this->next_layer_)(
buffered_read_some_handler<MutableBufferSequence,
typename decay<ReadHandler>::type>(
*storage, buffers, handler2.value),
storage);
}
}

private:
typename remove_reference<Stream>::type& next_layer_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <template <typename, typename> class Associator,
typename MutableBufferSequence, typename ReadHandler,
typename DefaultCandidate>
struct associator<Associator,
detail::buffered_read_some_handler<MutableBufferSequence, ReadHandler>,
DefaultCandidate>
: Associator<ReadHandler, DefaultCandidate>
{
static typename Associator<ReadHandler, DefaultCandidate>::type get(
const detail::buffered_read_some_handler<
MutableBufferSequence, ReadHandler>& h,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<ReadHandler, DefaultCandidate>::get(h.handler_, c);
}
};

#endif 

template <typename Stream>
template <typename MutableBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
buffered_read_stream<Stream>::async_read_some(
const MutableBufferSequence& buffers,
ASIO_MOVE_ARG(ReadHandler) handler)
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
detail::initiate_async_buffered_read_some<Stream>(next_layer_),
handler, &storage_, buffers);
}

template <typename Stream>
template <typename MutableBufferSequence>
std::size_t buffered_read_stream<Stream>::peek(
const MutableBufferSequence& buffers)
{
if (storage_.empty())
this->fill();
return this->peek_copy(buffers);
}

template <typename Stream>
template <typename MutableBufferSequence>
std::size_t buffered_read_stream<Stream>::peek(
const MutableBufferSequence& buffers, asio::error_code& ec)
{
ec = asio::error_code();
if (storage_.empty() && !this->fill(ec))
return 0;
return this->peek_copy(buffers);
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
