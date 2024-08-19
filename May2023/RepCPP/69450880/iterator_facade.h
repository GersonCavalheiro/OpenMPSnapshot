






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_facade_category.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/distance_from_result.h>

namespace hydra_thrust
{






template<typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference> class iterator_facade;



class iterator_core_access
{


template<typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference> friend class iterator_facade;

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator ==(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator !=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator <(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator >(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator <=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend bool
operator >=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
friend
typename hydra_thrust::detail::distance_from_result<
iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1>,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2>
>::type
operator-(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

template<typename Facade>
__host__ __device__
static typename Facade::reference dereference(Facade const& f)
{
return f.dereference();
}

template<typename Facade>
__host__ __device__
static void increment(Facade& f)
{
f.increment();
}

template<typename Facade>
__host__ __device__
static void decrement(Facade& f)
{
f.decrement();
}

template <class Facade1, class Facade2>
__host__ __device__
static bool equal(Facade1 const& f1, Facade2 const& f2)
{
return f1.equal(f2);
}



template <class Facade>
__host__ __device__
static void advance(Facade& f, typename Facade::difference_type n)
{
f.advance(n);
}

template <class Facade1, class Facade2>
__host__ __device__
static typename Facade1::difference_type
distance_from(Facade1 const& f1, Facade2 const& f2, hydra_thrust::detail::true_type)
{
return -f1.distance_to(f2);
}

template <class Facade1, class Facade2>
__host__ __device__
static typename Facade2::difference_type
distance_from(Facade1 const& f1, Facade2 const& f2, hydra_thrust::detail::false_type)
{
return f2.distance_to(f1);
}

template <class Facade1, class Facade2>
__host__ __device__
static typename hydra_thrust::detail::distance_from_result<Facade1,Facade2>::type
distance_from(Facade1 const& f1, Facade2 const& f2)
{
return distance_from(f1, f2,
typename hydra_thrust::detail::is_convertible<Facade2,Facade1>::type());
}

template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
__host__ __device__
static Derived& derived(iterator_facade<Derived,Value,System,Traversal,Reference,Difference>& facade)
{
return *static_cast<Derived*>(&facade);
}

template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
__host__ __device__
static Derived const& derived(iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& facade)
{
return *static_cast<Derived const*>(&facade);
}


}; 



template<typename Derived,
typename Value,
typename System,
typename Traversal,
typename Reference,
typename Difference = std::ptrdiff_t>
class iterator_facade
{
private:


__host__ __device__
Derived& derived()
{
return *static_cast<Derived*>(this);
}

__host__ __device__
Derived const& derived() const
{
return *static_cast<Derived const*>(this);
}


public:

typedef typename hydra_thrust::detail::remove_const<Value>::type value_type;


typedef Reference                                          reference;


typedef void                                               pointer;


typedef Difference                                         difference_type;


typedef typename hydra_thrust::detail::iterator_facade_category<
System, Traversal, Value, Reference
>::type                                                    iterator_category;


__host__ __device__
reference operator*() const
{
return iterator_core_access::dereference(this->derived());
}




__host__ __device__
reference operator[](difference_type n) const
{
return *(this->derived() + n);
}


__host__ __device__
Derived& operator++()
{
iterator_core_access::increment(this->derived());
return this->derived();
}


__host__ __device__
Derived  operator++(int)
{
Derived tmp(this->derived());
++*this;
return tmp;
}


__host__ __device__
Derived& operator--()
{
iterator_core_access::decrement(this->derived());
return this->derived();
}


__host__ __device__
Derived  operator--(int)
{
Derived tmp(this->derived());
--*this;
return tmp;
}


__host__ __device__
Derived& operator+=(difference_type n)
{
iterator_core_access::advance(this->derived(), n);
return this->derived();
}


__host__ __device__
Derived& operator-=(difference_type n)
{
iterator_core_access::advance(this->derived(), -n);
return this->derived();
}


__host__ __device__
Derived  operator-(difference_type n) const
{
Derived result(this->derived());
return result -= n;
}
}; 



template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator ==(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return iterator_core_access
::equal(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator !=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return !iterator_core_access
::equal(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator <(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return 0 > iterator_core_access
::distance_from(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator >(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return 0 < iterator_core_access
::distance_from(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator <=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return 0 >= iterator_core_access
::distance_from(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
bool
operator >=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return 0 <= iterator_core_access
::distance_from(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__

typename hydra_thrust::detail::distance_from_result<
iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1>,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2>
>::type

operator-(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
return iterator_core_access
::distance_from(*static_cast<Derived1 const*>(&lhs),
*static_cast<Derived2 const*>(&rhs));
}

template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
inline __host__ __device__
Derived operator+ (iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& i,
typename Derived::difference_type n)
{
Derived tmp(static_cast<Derived const&>(i));
return tmp += n;
}

template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
inline __host__ __device__
Derived operator+ (typename Derived::difference_type n,
iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& i)
{
Derived tmp(static_cast<Derived const&>(i));
return tmp += n;
}







} 

