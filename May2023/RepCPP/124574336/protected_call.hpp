



#ifndef BOOST_REGEX_V4_PROTECTED_CALL_HPP
#define BOOST_REGEX_V4_PROTECTED_CALL_HPP

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

namespace boost{
namespace BOOST_REGEX_DETAIL_NS{

class BOOST_REGEX_DECL abstract_protected_call
{
public:
bool BOOST_REGEX_CALL execute()const;
virtual ~abstract_protected_call(){}
private:
virtual bool call()const = 0;
};

template <class T>
class concrete_protected_call
: public abstract_protected_call
{
public:
typedef bool (T::*proc_type)();
concrete_protected_call(T* o, proc_type p)
: obj(o), proc(p) {}
private:
virtual bool call()const;
T* obj;
proc_type proc;
};

template <class T>
bool concrete_protected_call<T>::call()const
{
return (obj->*proc)();
}

}
} 

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
