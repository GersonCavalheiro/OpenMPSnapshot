#ifndef BOOST_STATECHART_FIFO_WORKER_HPP_INCLUDED
#define BOOST_STATECHART_FIFO_WORKER_HPP_INCLUDED



#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>
#include <boost/function/function0.hpp>
#include <boost/bind.hpp>
#include <boost/config.hpp>

#include <boost/detail/allocator_utilities.hpp>

#ifdef BOOST_HAS_THREADS
#  ifdef BOOST_MSVC
#    pragma warning( push )
#    pragma warning( disable: 4127 )
#    pragma warning( disable: 4244 )
#    pragma warning( disable: 4251 )
#    pragma warning( disable: 4512 )
#    pragma warning( disable: 4996 )
#  endif

#  include <boost/thread/mutex.hpp>
#  include <boost/thread/condition.hpp>

#  ifdef BOOST_MSVC
#    pragma warning( pop )
#  endif
#endif

#include <list>
#include <memory>   


namespace boost
{
namespace statechart
{



template< class Allocator = std::allocator< none > >
class fifo_worker : noncopyable
{
public:
#ifdef BOOST_HAS_THREADS
fifo_worker( bool waitOnEmptyQueue = false ) :
waitOnEmptyQueue_( waitOnEmptyQueue ),
#else
fifo_worker() :
#endif
terminated_( false )
{
}

typedef function0< void > work_item;

void queue_work_item( work_item & item )
{
if ( item.empty() )
{
return;
}

#ifdef BOOST_HAS_THREADS
mutex::scoped_lock lock( mutex_ );
#endif

workQueue_.push_back( work_item() );
workQueue_.back().swap( item );

#ifdef BOOST_HAS_THREADS
queueNotEmpty_.notify_one();
#endif
}

void queue_work_item( const work_item & item )
{
work_item copy = item;
queue_work_item( copy );
}

void terminate()
{
work_item item = boost::bind( &fifo_worker::terminate_impl, this );
queue_work_item( item );
}

bool terminated() const
{
return terminated_;
}

unsigned long operator()( unsigned long maxItemCount = 0 )
{
unsigned long itemCount = 0;

while ( !terminated() &&
( ( maxItemCount == 0 ) || ( itemCount < maxItemCount ) ) )
{
work_item item = dequeue_item();

if ( item.empty() )
{
return itemCount;
}

item();
++itemCount;
}

return itemCount;
}

private:
work_item dequeue_item()
{
#ifdef BOOST_HAS_THREADS
mutex::scoped_lock lock( mutex_ );

if ( !waitOnEmptyQueue_ && workQueue_.empty() )
{
return work_item();
}

while ( workQueue_.empty() )
{
queueNotEmpty_.wait( lock );
}
#else
if ( workQueue_.empty() )
{
return work_item();
}
#endif

work_item result;
result.swap( workQueue_.front() );
workQueue_.pop_front();
return result;
}

void terminate_impl()
{
terminated_ = true;
}


typedef std::list<
work_item,
typename boost::detail::allocator::rebind_to<
Allocator, work_item >::type
> work_queue_type;

work_queue_type workQueue_;

#ifdef BOOST_HAS_THREADS
mutex mutex_;
condition queueNotEmpty_;
const bool waitOnEmptyQueue_;
#endif

bool terminated_;
};



} 
} 



#endif
