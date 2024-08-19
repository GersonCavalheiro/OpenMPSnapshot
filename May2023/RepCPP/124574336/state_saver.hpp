#ifndef BOOST_SERIALIZATION_STATE_SAVER_HPP
#define BOOST_SERIALIZATION_STATE_SAVER_HPP

#if defined(_MSC_VER)
# pragma once
#endif






#include <boost/config.hpp>
#ifndef BOOST_NO_EXCEPTIONS
#include <exception>
#endif

#include <boost/call_traits.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits/has_nothrow_copy.hpp>
#include <boost/core/no_exceptions_support.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

namespace boost {
namespace serialization {

template<class T>
class state_saver : private boost::noncopyable
{
private:
const T previous_value;
T & previous_ref;

struct restore {
static void invoke(T & previous_ref, const T & previous_value){
previous_ref = previous_value; 
}
};

struct restore_with_exception {
static void invoke(T & previous_ref, const T & previous_value){
BOOST_TRY{
previous_ref = previous_value;
}
BOOST_CATCH(::std::exception &) {
}
BOOST_CATCH_END
}
};

public:
state_saver(
T & object
) :
previous_value(object),
previous_ref(object)
{}

~state_saver() {
#ifndef BOOST_NO_EXCEPTIONS
typedef typename mpl::eval_if<
has_nothrow_copy< T >,
mpl::identity<restore>,
mpl::identity<restore_with_exception>
>::type typex;
typex::invoke(previous_ref, previous_value);
#else
previous_ref = previous_value;
#endif
}

}; 

} 
} 

#endif 
