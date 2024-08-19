
#ifndef ASIO_DETAIL_THREAD_GROUP_HPP
#define ASIO_DETAIL_THREAD_GROUP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/thread.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class thread_group
{
public:
thread_group()
: first_(0)
{
}

~thread_group()
{
join();
}

template <typename Function>
void create_thread(Function f)
{
first_ = new item(f, first_);
}

template <typename Function>
void create_threads(Function f, std::size_t num_threads)
{
for (std::size_t i = 0; i < num_threads; ++i)
create_thread(f);
}

void join()
{
while (first_)
{
first_->thread_.join();
item* tmp = first_;
first_ = first_->next_;
delete tmp;
}
}

bool empty() const
{
return first_ == 0;
}

private:
struct item
{
template <typename Function>
explicit item(Function f, item* next)
: thread_(f),
next_(next)
{
}

asio::detail::thread thread_;
item* next_;
};

item* first_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
