

#ifndef BOOST_MULTI_INDEX_DETAIL_ITER_ADAPTOR_HPP
#define BOOST_MULTI_INDEX_DETAIL_ITER_ADAPTOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/apply.hpp>
#include <boost/operators.hpp>

namespace boost{

namespace multi_index{

namespace detail{





class iter_adaptor_access
{
public:
template<class Class>
static typename Class::reference dereference(const Class& x)
{
return x.dereference();
}

template<class Class>
static bool equal(const Class& x,const Class& y)
{
return x.equal(y);
}

template<class Class>
static void increment(Class& x)
{
x.increment();
}

template<class Class>
static void decrement(Class& x)
{
x.decrement();
}

template<class Class>
static void advance(Class& x,typename Class::difference_type n)
{
x.advance(n);
}

template<class Class>
static typename Class::difference_type distance_to(
const Class& x,const Class& y)
{
return x.distance_to(y);
}
};

template<typename Category>
struct iter_adaptor_selector;

template<class Derived,class Base>
class forward_iter_adaptor_base:
public forward_iterator_helper<
Derived,
typename Base::value_type,
typename Base::difference_type,
typename Base::pointer,
typename Base::reference>
{
public:
typedef typename Base::reference reference;

reference operator*()const
{
return iter_adaptor_access::dereference(final());
}

friend bool operator==(const Derived& x,const Derived& y)
{
return iter_adaptor_access::equal(x,y);
}

Derived& operator++()
{
iter_adaptor_access::increment(final());
return final();
}

private:
Derived& final(){return *static_cast<Derived*>(this);}
const Derived& final()const{return *static_cast<const Derived*>(this);}
};

template<class Derived,class Base>
bool operator==(
const forward_iter_adaptor_base<Derived,Base>& x,
const forward_iter_adaptor_base<Derived,Base>& y)
{
return iter_adaptor_access::equal(
static_cast<const Derived&>(x),static_cast<const Derived&>(y));
}

template<>
struct iter_adaptor_selector<std::forward_iterator_tag>
{
template<class Derived,class Base>
struct apply
{
typedef forward_iter_adaptor_base<Derived,Base> type;
};
};

template<class Derived,class Base>
class bidirectional_iter_adaptor_base:
public bidirectional_iterator_helper<
Derived,
typename Base::value_type,
typename Base::difference_type,
typename Base::pointer,
typename Base::reference>
{
public:
typedef typename Base::reference reference;

reference operator*()const
{
return iter_adaptor_access::dereference(final());
}

friend bool operator==(const Derived& x,const Derived& y)
{
return iter_adaptor_access::equal(x,y);
}

Derived& operator++()
{
iter_adaptor_access::increment(final());
return final();
}

Derived& operator--()
{
iter_adaptor_access::decrement(final());
return final();
}

private:
Derived& final(){return *static_cast<Derived*>(this);}
const Derived& final()const{return *static_cast<const Derived*>(this);}
};

template<class Derived,class Base>
bool operator==(
const bidirectional_iter_adaptor_base<Derived,Base>& x,
const bidirectional_iter_adaptor_base<Derived,Base>& y)
{
return iter_adaptor_access::equal(
static_cast<const Derived&>(x),static_cast<const Derived&>(y));
}

template<>
struct iter_adaptor_selector<std::bidirectional_iterator_tag>
{
template<class Derived,class Base>
struct apply
{
typedef bidirectional_iter_adaptor_base<Derived,Base> type;
};
};

template<class Derived,class Base>
class random_access_iter_adaptor_base:
public random_access_iterator_helper<
Derived,
typename Base::value_type,
typename Base::difference_type,
typename Base::pointer,
typename Base::reference>
{
public:
typedef typename Base::reference       reference;
typedef typename Base::difference_type difference_type;

reference operator*()const
{
return iter_adaptor_access::dereference(final());
}

friend bool operator==(const Derived& x,const Derived& y)
{
return iter_adaptor_access::equal(x,y);
}

friend bool operator<(const Derived& x,const Derived& y)
{
return iter_adaptor_access::distance_to(x,y)>0;
}

Derived& operator++()
{
iter_adaptor_access::increment(final());
return final();
}

Derived& operator--()
{
iter_adaptor_access::decrement(final());
return final();
}

Derived& operator+=(difference_type n)
{
iter_adaptor_access::advance(final(),n);
return final();
}

Derived& operator-=(difference_type n)
{
iter_adaptor_access::advance(final(),-n);
return final();
}

friend difference_type operator-(const Derived& x,const Derived& y)
{
return iter_adaptor_access::distance_to(y,x);
}

private:
Derived& final(){return *static_cast<Derived*>(this);}
const Derived& final()const{return *static_cast<const Derived*>(this);}
};

template<class Derived,class Base>
bool operator==(
const random_access_iter_adaptor_base<Derived,Base>& x,
const random_access_iter_adaptor_base<Derived,Base>& y)
{
return iter_adaptor_access::equal(
static_cast<const Derived&>(x),static_cast<const Derived&>(y));
}

template<class Derived,class Base>
bool operator<(
const random_access_iter_adaptor_base<Derived,Base>& x,
const random_access_iter_adaptor_base<Derived,Base>& y)
{
return iter_adaptor_access::distance_to(
static_cast<const Derived&>(x),static_cast<const Derived&>(y))>0;
}

template<class Derived,class Base>
typename random_access_iter_adaptor_base<Derived,Base>::difference_type
operator-(
const random_access_iter_adaptor_base<Derived,Base>& x,
const random_access_iter_adaptor_base<Derived,Base>& y)
{
return iter_adaptor_access::distance_to(
static_cast<const Derived&>(y),static_cast<const Derived&>(x));
}

template<>
struct iter_adaptor_selector<std::random_access_iterator_tag>
{
template<class Derived,class Base>
struct apply
{
typedef random_access_iter_adaptor_base<Derived,Base> type;
};
};

template<class Derived,class Base>
struct iter_adaptor_base
{
typedef iter_adaptor_selector<
typename Base::iterator_category> selector;
typedef typename mpl::apply2<
selector,Derived,Base>::type      type;
};

template<class Derived,class Base>
class iter_adaptor:public iter_adaptor_base<Derived,Base>::type
{
protected:
iter_adaptor(){}
explicit iter_adaptor(const Base& b_):b(b_){}

const Base& base_reference()const{return b;}
Base&       base_reference(){return b;}

private:
Base b;
};

} 

} 

} 

#endif
