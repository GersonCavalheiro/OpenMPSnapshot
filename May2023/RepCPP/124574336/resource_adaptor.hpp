
#ifndef BOOST_CONTAINER_PMR_RESOURCE_ADAPTOR_HPP
#define BOOST_CONTAINER_PMR_RESOURCE_ADAPTOR_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/container_fwd.hpp>

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/allocator_traits.hpp>
#include <boost/intrusive/detail/ebo_functor_holder.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/detail/type_traits.hpp>
#include <boost/container/detail/std_fwd.hpp>

#include <cstring>

namespace boost {
namespace container {

namespace pmr_dtl {

template<class T>
struct max_allocator_alignment
{
static const std::size_t value = 1;
};

template<class T>
struct max_allocator_alignment< ::boost::container::new_allocator<T> >
{
static const std::size_t value = boost::move_detail::alignment_of<boost::move_detail::max_align_t>::value;
};

template<class T>
struct max_allocator_alignment< std::allocator<T> >
{
static const std::size_t value = boost::move_detail::alignment_of<boost::move_detail::max_align_t>::value;
};

}  

namespace pmr {

template <class Allocator>
class resource_adaptor_imp
: public  memory_resource
#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED
, private ::boost::intrusive::detail::ebo_functor_holder<Allocator>
#endif
{
#ifdef BOOST_CONTAINER_DOXYGEN_INVOKED
Allocator m_alloc;
#else
BOOST_COPYABLE_AND_MOVABLE(resource_adaptor_imp)
typedef ::boost::intrusive::detail::ebo_functor_holder<Allocator> ebo_alloc_t;
void static_assert_if_not_char_allocator() const
{
BOOST_STATIC_ASSERT((boost::container::dtl::is_same<typename Allocator::value_type, char>::value));
}
#endif

public:
typedef Allocator allocator_type;

resource_adaptor_imp()
{  this->static_assert_if_not_char_allocator(); }

resource_adaptor_imp(const resource_adaptor_imp &other)
: ebo_alloc_t(other.ebo_alloc_t::get())
{}

resource_adaptor_imp(BOOST_RV_REF(resource_adaptor_imp) other)
: ebo_alloc_t(::boost::move(other.get()))
{}

explicit resource_adaptor_imp(const Allocator& a2)
: ebo_alloc_t(a2)
{  this->static_assert_if_not_char_allocator(); }

explicit resource_adaptor_imp(BOOST_RV_REF(Allocator) a2)
: ebo_alloc_t(::boost::move(a2))
{  this->static_assert_if_not_char_allocator(); }

resource_adaptor_imp& operator=(BOOST_COPY_ASSIGN_REF(resource_adaptor_imp) other)
{  this->ebo_alloc_t::get() = other.ebo_alloc_t::get(); return *this;  }

resource_adaptor_imp& operator=(BOOST_RV_REF(resource_adaptor_imp) other)
{  this->ebo_alloc_t::get() = ::boost::move(other.ebo_alloc_t::get()); return *this;  }

allocator_type &get_allocator()
{  return this->ebo_alloc_t::get(); }

const allocator_type &get_allocator() const
{  return this->ebo_alloc_t::get(); }

protected:
virtual void* do_allocate(std::size_t bytes, std::size_t alignment)
{
if (alignment <= priv_guaranteed_allocator_alignment())
return this->ebo_alloc_t::get().allocate(bytes);
else
return this->priv_aligned_alloc(bytes, alignment);
}

virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment)
{
if (alignment <= priv_guaranteed_allocator_alignment())
this->ebo_alloc_t::get().deallocate((char*)p, bytes);
else
this->priv_aligned_dealloc(p, bytes, alignment);
}

virtual bool do_is_equal(const memory_resource& other) const BOOST_NOEXCEPT
{
const resource_adaptor_imp* p = dynamic_cast<const resource_adaptor_imp*>(&other);
return p && p->ebo_alloc_t::get() == this->ebo_alloc_t::get();
}

private:
void * priv_aligned_alloc(std::size_t bytes, std::size_t alignment)
{
void *const p = this->ebo_alloc_t::get().allocate(bytes + priv_extra_bytes_for_overalignment(alignment));

if (0 != p) {
void *const aligned_ptr = (void*)(((std::size_t)p + priv_extra_bytes_for_overalignment(alignment)) & ~(alignment - 1));

std::memcpy(priv_bookeeping_addr_from_aligned_ptr(aligned_ptr), &p, sizeof(p));
return aligned_ptr;
}
return 0;
}

void priv_aligned_dealloc(void *aligned_ptr, std::size_t bytes, std::size_t alignment)
{
void *p;
std::memcpy(&p, priv_bookeeping_addr_from_aligned_ptr(aligned_ptr), sizeof(p));
std::size_t s  = bytes + priv_extra_bytes_for_overalignment(alignment);
this->ebo_alloc_t::get().deallocate((char*)p, s);
}

static BOOST_CONTAINER_FORCEINLINE void *priv_bookeeping_addr_from_aligned_ptr(void *aligned_ptr)
{
return reinterpret_cast<void*>(reinterpret_cast<std::size_t>(aligned_ptr) - sizeof(void*));
}

BOOST_CONTAINER_FORCEINLINE static std::size_t priv_extra_bytes_for_overalignment(std::size_t alignment)
{
return alignment - 1 + sizeof(void*);
}

BOOST_CONTAINER_FORCEINLINE static std::size_t priv_guaranteed_allocator_alignment()
{
return pmr_dtl::max_allocator_alignment<Allocator>::value;
}
};

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

template <class Allocator>
using resource_adaptor = resource_adaptor_imp
<typename allocator_traits<Allocator>::template rebind_alloc<char> >;

#else

template <class Allocator>
class resource_adaptor
: public resource_adaptor_imp
<typename allocator_traits<Allocator>::template portable_rebind_alloc<char>::type>
{
typedef resource_adaptor_imp
<typename allocator_traits<Allocator>::template portable_rebind_alloc<char>::type> base_t;

BOOST_COPYABLE_AND_MOVABLE(resource_adaptor)

public:
resource_adaptor()
: base_t()
{}

resource_adaptor(const resource_adaptor &other)
: base_t(other)
{}

resource_adaptor(BOOST_RV_REF(resource_adaptor) other)
: base_t(BOOST_MOVE_BASE(base_t, other))
{}

explicit resource_adaptor(const Allocator& a2)
: base_t(a2)
{}

explicit resource_adaptor(BOOST_RV_REF(Allocator) a2)
: base_t(::boost::move(a2))
{}

resource_adaptor& operator=(BOOST_COPY_ASSIGN_REF(resource_adaptor) other)
{  return static_cast<resource_adaptor&>(this->base_t::operator=(other));  }

resource_adaptor& operator=(BOOST_RV_REF(resource_adaptor) other)
{  return static_cast<resource_adaptor&>(this->base_t::operator=(BOOST_MOVE_BASE(base_t, other)));  }

};

#endif

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
