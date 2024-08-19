

#ifndef BOOST_MULTI_INDEX_DETAIL_MODIFY_KEY_ADAPTOR_HPP
#define BOOST_MULTI_INDEX_DETAIL_MODIFY_KEY_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost{

namespace multi_index{

namespace detail{



template<typename Fun,typename Value,typename KeyFromValue>
struct modify_key_adaptor
{

modify_key_adaptor(Fun f_,KeyFromValue kfv_):f(f_),kfv(kfv_){}

void operator()(Value& x)
{
f(kfv(x));
}

private:
Fun          f;
KeyFromValue kfv;
};

} 

} 

} 

#endif
