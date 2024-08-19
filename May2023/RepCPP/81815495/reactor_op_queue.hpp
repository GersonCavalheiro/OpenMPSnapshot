
#ifndef ASIO_DETAIL_REACTOR_OP_QUEUE_HPP
#define ASIO_DETAIL_REACTOR_OP_QUEUE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Descriptor>
class reactor_op_queue
: private noncopyable
{
public:
typedef Descriptor key_type;

struct mapped_type : op_queue<reactor_op>
{
mapped_type() {}
mapped_type(const mapped_type&) {}
void operator=(const mapped_type&) {}
};

typedef typename hash_map<key_type, mapped_type>::value_type value_type;
typedef typename hash_map<key_type, mapped_type>::iterator iterator;

reactor_op_queue()
: operations_()
{
}

iterator begin() { return operations_.begin(); }
iterator end() { return operations_.end(); }

bool enqueue_operation(Descriptor descriptor, reactor_op* op)
{
std::pair<iterator, bool> entry =
operations_.insert(value_type(descriptor, mapped_type()));
entry.first->second.push(op);
return entry.second;
}

bool cancel_operations(iterator i, op_queue<operation>& ops,
const asio::error_code& ec =
asio::error::operation_aborted)
{
if (i != operations_.end())
{
while (reactor_op* op = i->second.front())
{
op->ec_ = ec;
i->second.pop();
ops.push(op);
}
operations_.erase(i);
return true;
}

return false;
}

bool cancel_operations(Descriptor descriptor, op_queue<operation>& ops,
const asio::error_code& ec =
asio::error::operation_aborted)
{
return this->cancel_operations(operations_.find(descriptor), ops, ec);
}

bool cancel_operations_by_key(iterator i, op_queue<operation>& ops,
void* cancellation_key, const asio::error_code& ec =
asio::error::operation_aborted)
{
bool result = false;
if (i != operations_.end())
{
op_queue<reactor_op> other_ops;
while (reactor_op* op = i->second.front())
{
i->second.pop();
if (op->cancellation_key_ == cancellation_key)
{
op->ec_ = ec;
ops.push(op);
result = true;
}
else
other_ops.push(op);
}
i->second.push(other_ops);
if (i->second.empty())
operations_.erase(i);
}
return result;
}

bool cancel_operations_by_key(Descriptor descriptor, op_queue<operation>& ops,
void* cancellation_key, const asio::error_code& ec =
asio::error::operation_aborted)
{
return this->cancel_operations_by_key(
operations_.find(descriptor), ops, cancellation_key, ec);
}

bool empty() const
{
return operations_.empty();
}

bool has_operation(Descriptor descriptor) const
{
return operations_.find(descriptor) != operations_.end();
}

bool perform_operations(iterator i, op_queue<operation>& ops)
{
if (i != operations_.end())
{
while (reactor_op* op = i->second.front())
{
if (op->perform())
{
i->second.pop();
ops.push(op);
}
else
{
return true;
}
}
operations_.erase(i);
}
return false;
}

bool perform_operations(Descriptor descriptor, op_queue<operation>& ops)
{
return this->perform_operations(operations_.find(descriptor), ops);
}

void get_all_operations(op_queue<operation>& ops)
{
iterator i = operations_.begin();
while (i != operations_.end())
{
iterator op_iter = i++;
ops.push(op_iter->second);
operations_.erase(op_iter);
}
}

private:
hash_map<key_type, mapped_type> operations_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
