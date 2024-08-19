#ifndef BOOST_STATECHART_DETAIL_STATE_BASE_HPP_INCLUDED
#define BOOST_STATECHART_DETAIL_STATE_BASE_HPP_INCLUDED



#include <boost/statechart/result.hpp>
#include <boost/statechart/event.hpp>

#include <boost/statechart/detail/counted_base.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <boost/assert.hpp>
#include <boost/config.hpp> 

#include <boost/detail/workaround.hpp>
#include <boost/detail/allocator_utilities.hpp>

#ifdef BOOST_MSVC
#  pragma warning( push )
#  pragma warning( disable: 4702 ) 
#endif

#include <list>

#ifdef BOOST_MSVC
#  pragma warning( pop )
#endif



namespace boost
{
namespace statechart
{
namespace detail
{



template< class Allocator, class RttiPolicy >
class leaf_state;
template< class Allocator, class RttiPolicy >
class node_state_base;

typedef unsigned char orthogonal_position_type;



template< class Allocator, class RttiPolicy >
class state_base :
#ifndef NDEBUG
noncopyable,
#endif
public RttiPolicy::template rtti_base_type<
counted_base< false > >
{
typedef typename RttiPolicy::template rtti_base_type<
counted_base< false > > base_type;

public:
void exit() {}

virtual const state_base * outer_state_ptr() const = 0;

protected:
state_base( typename RttiPolicy::id_provider_type idProvider ) :
base_type( idProvider ),
deferredEvents_( false )
{
}

#if BOOST_WORKAROUND( __GNUC__, BOOST_TESTED_AT( 4 ) )
virtual ~state_base() {}
#else
~state_base() {}
#endif

protected:
void defer_event()
{
deferredEvents_ = true;
}

bool deferred_events() const
{
return deferredEvents_;
}

template< class Context >
void set_context( orthogonal_position_type position, Context * pContext )
{
pContext->add_inner_state( position, this );
}

public:
virtual detail::reaction_result react_impl(
const event_base & evt,
typename RttiPolicy::id_type eventType ) = 0;

typedef intrusive_ptr< node_state_base< Allocator, RttiPolicy > >
node_state_base_ptr_type;
typedef intrusive_ptr< leaf_state< Allocator, RttiPolicy > >
leaf_state_ptr_type;
typedef std::list<
leaf_state_ptr_type,
typename boost::detail::allocator::rebind_to<
Allocator, leaf_state_ptr_type >::type
> state_list_type;

virtual void remove_from_state_list(
typename state_list_type::iterator & statesEnd,
node_state_base_ptr_type & pOutermostUnstableState,
bool performFullExit ) = 0;

private:
bool deferredEvents_;
};



#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
} 
} 
#endif



template< class Allocator, class RttiPolicy >
inline void intrusive_ptr_add_ref(
const ::boost::statechart::detail::state_base< Allocator, RttiPolicy > * pBase )
{
pBase->add_ref();
}

template< class Allocator, class RttiPolicy >
inline void intrusive_ptr_release( 
const ::boost::statechart::detail::state_base< Allocator, RttiPolicy > * pBase )
{
if ( pBase->release() )
{
BOOST_ASSERT( false );
}
}



#ifndef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
} 
} 
#endif



} 



#endif
