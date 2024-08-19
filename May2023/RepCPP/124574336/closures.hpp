
#ifndef BOOST_SPIRIT_CLASSIC_PHOENIX_CLOSURES_HPP
#define BOOST_SPIRIT_CLASSIC_PHOENIX_CLOSURES_HPP

#include <boost/spirit/home/classic/phoenix/actor.hpp>
#include <boost/assert.hpp>

#ifdef PHOENIX_THREADSAFE
#include <boost/thread/tss.hpp>
#include <boost/thread/once.hpp>
#endif

namespace phoenix {

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif


namespace impl
{
#ifndef PHOENIX_THREADSAFE
template <typename FrameT>
struct closure_frame_holder
{
typedef FrameT frame_t;
typedef frame_t *frame_ptr;

closure_frame_holder() : frame(0) {}

frame_ptr &get() { return frame; }
void set(frame_t *f) { frame = f; }

private:
frame_ptr frame;

closure_frame_holder(closure_frame_holder const &);
closure_frame_holder &operator=(closure_frame_holder const &);
};
#else
template <typename FrameT>
struct closure_frame_holder
{
typedef FrameT   frame_t;
typedef frame_t *frame_ptr;

closure_frame_holder() : tsp_frame() {}

frame_ptr &get()
{
if (!tsp_frame.get())
tsp_frame.reset(new frame_ptr(0));
return *tsp_frame;
}
void set(frame_ptr f)
{
*tsp_frame = f;
}

private:
boost::thread_specific_ptr<frame_ptr> tsp_frame;

closure_frame_holder(closure_frame_holder const &);
closure_frame_holder &operator=(closure_frame_holder const &);
};
#endif
} 

template <typename ClosureT>
class closure_frame : public ClosureT::tuple_t {

public:

closure_frame(ClosureT const& clos)
: ClosureT::tuple_t(), save(clos.frame.get()), frame(clos.frame)
{ clos.frame.set(this); }

template <typename TupleT>
closure_frame(ClosureT const& clos, TupleT const& init)
: ClosureT::tuple_t(init), save(clos.frame.get()), frame(clos.frame)
{ clos.frame.set(this); }

~closure_frame()
{ frame.set(save); }

private:

closure_frame(closure_frame const&);            
closure_frame& operator=(closure_frame const&); 

closure_frame* save;
impl::closure_frame_holder<closure_frame>& frame;
};

template <int N, typename ClosureT>
class closure_member {

public:

typedef typename ClosureT::tuple_t tuple_t;

closure_member()
: frame(ClosureT::closure_frame_holder_ref()) {}

template <typename TupleT>
struct result {

typedef typename tuple_element<
N, typename ClosureT::tuple_t
>::rtype type;
};

template <typename TupleT>
typename tuple_element<N, typename ClosureT::tuple_t>::rtype
eval(TupleT const& ) const
{
using namespace std;
BOOST_ASSERT(frame.get() != 0);
return (*frame.get())[tuple_index<N>()];
}

private:
impl::closure_frame_holder<typename ClosureT::closure_frame_t> &frame;
};

template <
typename T0 = nil_t
,   typename T1 = nil_t
,   typename T2 = nil_t

#if PHOENIX_LIMIT > 3
,   typename T3 = nil_t
,   typename T4 = nil_t
,   typename T5 = nil_t

#if PHOENIX_LIMIT > 6
,   typename T6 = nil_t
,   typename T7 = nil_t
,   typename T8 = nil_t

#if PHOENIX_LIMIT > 9
,   typename T9 = nil_t
,   typename T10 = nil_t
,   typename T11 = nil_t

#if PHOENIX_LIMIT > 12
,   typename T12 = nil_t
,   typename T13 = nil_t
,   typename T14 = nil_t

#endif
#endif
#endif
#endif
>
class closure {

public:

typedef tuple<
T0, T1, T2
#if PHOENIX_LIMIT > 3
,   T3, T4, T5
#if PHOENIX_LIMIT > 6
,   T6, T7, T8
#if PHOENIX_LIMIT > 9
,   T9, T10, T11
#if PHOENIX_LIMIT > 12
,   T12, T13, T14
#endif
#endif
#endif
#endif
> tuple_t;

typedef closure<
T0, T1, T2
#if PHOENIX_LIMIT > 3
,   T3, T4, T5
#if PHOENIX_LIMIT > 6
,   T6, T7, T8
#if PHOENIX_LIMIT > 9
,   T9, T10, T11
#if PHOENIX_LIMIT > 12
,   T12, T13, T14
#endif
#endif
#endif
#endif
> self_t;

typedef closure_frame<self_t> closure_frame_t;

closure()
: frame()       { closure_frame_holder_ref(&frame); }

typedef actor<closure_member<0, self_t> > member1;
typedef actor<closure_member<1, self_t> > member2;
typedef actor<closure_member<2, self_t> > member3;

#if PHOENIX_LIMIT > 3
typedef actor<closure_member<3, self_t> > member4;
typedef actor<closure_member<4, self_t> > member5;
typedef actor<closure_member<5, self_t> > member6;

#if PHOENIX_LIMIT > 6
typedef actor<closure_member<6, self_t> > member7;
typedef actor<closure_member<7, self_t> > member8;
typedef actor<closure_member<8, self_t> > member9;

#if PHOENIX_LIMIT > 9
typedef actor<closure_member<9, self_t> > member10;
typedef actor<closure_member<10, self_t> > member11;
typedef actor<closure_member<11, self_t> > member12;

#if PHOENIX_LIMIT > 12
typedef actor<closure_member<12, self_t> > member13;
typedef actor<closure_member<13, self_t> > member14;
typedef actor<closure_member<14, self_t> > member15;

#endif
#endif
#endif
#endif

#if !defined(__MWERKS__) || (__MWERKS__ > 0x3002)
private:
#endif

closure(closure const&);            
closure& operator=(closure const&); 

#if !defined(__MWERKS__) || (__MWERKS__ > 0x3002)
template <int N, typename ClosureT>
friend class closure_member;

template <typename ClosureT>
friend class closure_frame;
#endif

typedef impl::closure_frame_holder<closure_frame_t> holder_t;

#ifdef PHOENIX_THREADSAFE
static boost::thread_specific_ptr<holder_t*> &
tsp_frame_instance()
{
static boost::thread_specific_ptr<holder_t*> the_instance;
return the_instance;
}

static void
tsp_frame_instance_init()
{
tsp_frame_instance();
}
#endif

static holder_t &
closure_frame_holder_ref(holder_t* holder_ = 0)
{
#ifdef PHOENIX_THREADSAFE
#ifndef BOOST_THREAD_PROVIDES_ONCE_CXX11
static boost::once_flag been_here = BOOST_ONCE_INIT;
#else
static boost::once_flag been_here;
#endif
boost::call_once(been_here, tsp_frame_instance_init);
boost::thread_specific_ptr<holder_t*> &tsp_frame = tsp_frame_instance();
if (!tsp_frame.get())
tsp_frame.reset(new holder_t *(0));
holder_t *& holder = *tsp_frame;
#else
static holder_t* holder = 0;
#endif
if (holder_ != 0)
holder = holder_;
return *holder;
}

mutable holder_t frame;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif

}

#endif
