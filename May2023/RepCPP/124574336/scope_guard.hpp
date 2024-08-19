


#ifndef BOOST_SIGNALS2_DETAIL_SCOPE_GUARD_HPP
#define BOOST_SIGNALS2_DETAIL_SCOPE_GUARD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/core/no_exceptions_support.hpp>

namespace boost{

namespace signals2{

namespace detail{



class scope_guard_impl_base
{
public:
scope_guard_impl_base():dismissed_(false){}
void dismiss()const{dismissed_=true;}

protected:
~scope_guard_impl_base(){}

scope_guard_impl_base(const scope_guard_impl_base& other):
dismissed_(other.dismissed_)
{
other.dismiss();
}

template<typename J>
static void safe_execute(J& j){
BOOST_TRY{
if(!j.dismissed_)j.execute();
}
BOOST_CATCH(...){}
BOOST_CATCH_END
}

mutable bool dismissed_;

private:
scope_guard_impl_base& operator=(const scope_guard_impl_base&);
};

typedef const scope_guard_impl_base& scope_guard;

template<class Obj,typename MemFun,typename P1,typename P2>
class obj_scope_guard_impl2:public scope_guard_impl_base
{
public:
obj_scope_guard_impl2(Obj& obj,MemFun mem_fun,P1 p1,P2 p2):
obj_(obj),mem_fun_(mem_fun),p1_(p1),p2_(p2)
{}
~obj_scope_guard_impl2(){scope_guard_impl_base::safe_execute(*this);}
void execute(){(obj_.*mem_fun_)(p1_,p2_);}

protected:
Obj&     obj_;
MemFun   mem_fun_;
const P1 p1_;
const P2 p2_;
};

template<class Obj,typename MemFun,typename P1,typename P2>
inline obj_scope_guard_impl2<Obj,MemFun,P1,P2>
make_obj_guard(Obj& obj,MemFun mem_fun,P1 p1,P2 p2)
{
return obj_scope_guard_impl2<Obj,MemFun,P1,P2>(obj,mem_fun,p1,p2);
}

} 

} 

} 

#endif
