
#ifndef BOOST_CONTRACT_DETAIL_CHECKING_HPP_
#define BOOST_CONTRACT_DETAIL_CHECKING_HPP_


#include <boost/contract/core/config.hpp>
#include <boost/contract/detail/static_local_var.hpp>
#include <boost/contract/detail/declspec.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/noncopyable.hpp>
#include <boost/config.hpp>

namespace boost { namespace contract { namespace detail {

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4275) 
#pragma warning(disable: 4251) 
#endif

class BOOST_CONTRACT_DETAIL_DECLSPEC checking :
private boost::noncopyable 
{
public:
explicit checking() {
#ifndef BOOST_CONTRACT_DISABLE_THREADS
init_locked();
#else
init_unlocked();
#endif
}

~checking() {
#ifndef BOOST_CONTRACT_DISABLE_THREADS
done_locked();
#else
done_unlocked();
#endif
}

static bool already() {
#ifndef BOOST_CONTRACT_DISABLE_THREADS
return already_locked();
#else
return already_unlocked();
#endif
}

private:
void init_unlocked();
void init_locked();

void done_unlocked();
void done_locked();

static bool already_unlocked();
static bool already_locked();

struct mutex_tag;
typedef static_local_var<mutex_tag, boost::mutex> mutex;

struct checking_tag;
typedef static_local_var_init<checking_tag, bool, bool, false> flag;
};

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

} } } 

#ifdef BOOST_CONTRACT_HEADER_ONLY
#include <boost/contract/detail/inlined/detail/checking.hpp>
#endif

#endif 

