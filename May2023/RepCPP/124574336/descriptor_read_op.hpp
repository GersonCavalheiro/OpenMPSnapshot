
#ifndef BOOST_ASIO_DETAIL_DESCRIPTOR_READ_OP_HPP
#define BOOST_ASIO_DETAIL_DESCRIPTOR_READ_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_WINDOWS) && !defined(__CYGWIN__)

#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/descriptor_ops.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/reactor_op.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename MutableBufferSequence>
class descriptor_read_op_base : public reactor_op
{
public:
descriptor_read_op_base(const boost::system::error_code& success_ec,
int descriptor, const MutableBufferSequence& buffers,
func_type complete_func)
: reactor_op(success_ec,
&descriptor_read_op_base::do_perform, complete_func),
descriptor_(descriptor),
buffers_(buffers)
{
}

static status do_perform(reactor_op* base)
{
descriptor_read_op_base* o(static_cast<descriptor_read_op_base*>(base));

typedef buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence> bufs_type;

status result;
if (bufs_type::is_single_buffer)
{
result = descriptor_ops::non_blocking_read1(o->descriptor_,
bufs_type::first(o->buffers_).data(),
bufs_type::first(o->buffers_).size(),
o->ec_, o->bytes_transferred_) ? done : not_done;
}
else
{
bufs_type bufs(o->buffers_);
result = descriptor_ops::non_blocking_read(o->descriptor_,
bufs.buffers(), bufs.count(), o->ec_, o->bytes_transferred_)
? done : not_done;
}

BOOST_ASIO_HANDLER_REACTOR_OPERATION((*o, "non_blocking_read",
o->ec_, o->bytes_transferred_));

return result;
}

private:
int descriptor_;
MutableBufferSequence buffers_;
};

template <typename MutableBufferSequence, typename Handler, typename IoExecutor>
class descriptor_read_op
: public descriptor_read_op_base<MutableBufferSequence>
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(descriptor_read_op);

descriptor_read_op(const boost::system::error_code& success_ec,
int descriptor, const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
: descriptor_read_op_base<MutableBufferSequence>(success_ec,
descriptor, buffers, &descriptor_read_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& ,
std::size_t )
{
descriptor_read_op* o(static_cast<descriptor_read_op*>(base));
ptr p = { boost::asio::detail::addressof(o->handler_), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder2<Handler, boost::system::error_code, std::size_t>
handler(o->handler_, o->ec_, o->bytes_transferred_);
p.h = boost::asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
w.complete(handler, handler.handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
