
#ifndef BOOST_XPRESSIVE_DETAIL_UTILITY_TRACKING_PTR_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_UTILITY_TRACKING_PTR_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#ifdef BOOST_XPRESSIVE_DEBUG_TRACKING_POINTER
# include <iostream>
#endif
#include <set>
#include <functional>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Type>
struct tracking_ptr;

template<typename Derived>
struct enable_reference_tracking;

template<typename Derived>
struct weak_iterator
: iterator_facade
<
weak_iterator<Derived>
, shared_ptr<Derived> const
, std::forward_iterator_tag
>
{
typedef std::set<weak_ptr<Derived> > set_type;
typedef typename set_type::iterator base_iterator;

weak_iterator()
: cur_()
, iter_()
, set_(0)
{
}

weak_iterator(base_iterator iter, set_type *set)
: cur_()
, iter_(iter)
, set_(set)
{
this->satisfy_();
}

private:
friend class boost::iterator_core_access;

shared_ptr<Derived> const &dereference() const
{
return this->cur_;
}

void increment()
{
++this->iter_;
this->satisfy_();
}

bool equal(weak_iterator<Derived> const &that) const
{
return this->iter_ == that.iter_;
}

void satisfy_()
{
while(this->iter_ != this->set_->end())
{
this->cur_ = this->iter_->lock();
if(this->cur_)
return;
base_iterator tmp = this->iter_++;
this->set_->erase(tmp);
}
this->cur_.reset();
}

shared_ptr<Derived> cur_;
base_iterator iter_;
set_type *set_;
};

template<typename Derived>
struct filter_self
{
typedef shared_ptr<Derived> argument_type;
typedef bool result_type;

filter_self(enable_reference_tracking<Derived> *self)
: self_(self)
{
}

bool operator ()(shared_ptr<Derived> const &that) const
{
return this->self_ != that.get();
}

private:
enable_reference_tracking<Derived> *self_;
};

template<typename T>
void adl_swap(T &t1, T &t2)
{
swap(t1, t2);
}

template<typename Derived>
struct enable_reference_tracking
{
typedef std::set<shared_ptr<Derived> > references_type;
typedef std::set<weak_ptr<Derived> > dependents_type;

void tracking_copy(Derived const &that)
{
if(&this->derived_() != &that)
{
this->raw_copy_(that);
this->tracking_update();
}
}

void tracking_clear()
{
this->raw_copy_(Derived());
}

void tracking_update()
{
this->update_references_();
this->update_dependents_();
}

void track_reference(enable_reference_tracking<Derived> &that)
{
that.purge_stale_deps_();
this->refs_.insert(that.self_);
this->refs_.insert(that.refs_.begin(), that.refs_.end());
}

long use_count() const
{
return this->cnt_;
}

void add_ref()
{
++this->cnt_;
}

void release()
{
BOOST_ASSERT(0 < this->cnt_);
if(0 == --this->cnt_)
{
this->refs_.clear();
this->self_.reset();
}
}

#ifdef BOOST_XPRESSIVE_DEBUG_TRACKING_POINTER
friend std::ostream &operator <<(std::ostream &sout, enable_reference_tracking<Derived> const &that)
{
that.dump_(sout);
return sout;
}
#endif

protected:

enable_reference_tracking()
: refs_()
, deps_()
, self_()
, cnt_(0)
{
}

enable_reference_tracking(enable_reference_tracking<Derived> const &that)
: refs_()
, deps_()
, self_()
, cnt_(0)
{
this->operator =(that);
}

enable_reference_tracking<Derived> &operator =(enable_reference_tracking<Derived> const &that)
{
references_type(that.refs_).swap(this->refs_);
return *this;
}

void swap(enable_reference_tracking<Derived> &that)
{
this->refs_.swap(that.refs_);
}

private:
friend struct tracking_ptr<Derived>;

Derived &derived_()
{
return *static_cast<Derived *>(this);
}

void raw_copy_(Derived that)
{
detail::adl_swap(this->derived_(), that);
}

bool has_deps_() const
{
return !this->deps_.empty();
}

void update_references_()
{
typename references_type::iterator cur = this->refs_.begin();
typename references_type::iterator end = this->refs_.end();
for(; cur != end; ++cur)
{
(*cur)->track_dependency_(*this);
}
}

void update_dependents_()
{
weak_iterator<Derived> cur(this->deps_.begin(), &this->deps_);
weak_iterator<Derived> end(this->deps_.end(), &this->deps_);

for(; cur != end; ++cur)
{
(*cur)->track_reference(*this);
}
}

void track_dependency_(enable_reference_tracking<Derived> &dep)
{
if(this == &dep) 
return;

this->deps_.insert(dep.self_);

filter_self<Derived> not_self(this);
weak_iterator<Derived> begin(dep.deps_.begin(), &dep.deps_);
weak_iterator<Derived> end(dep.deps_.end(), &dep.deps_);

this->deps_.insert(
make_filter_iterator(not_self, begin, end)
, make_filter_iterator(not_self, end, end)
);
}

void purge_stale_deps_()
{
weak_iterator<Derived> cur(this->deps_.begin(), &this->deps_);
weak_iterator<Derived> end(this->deps_.end(), &this->deps_);

for(; cur != end; ++cur)
;
}

#ifdef BOOST_XPRESSIVE_DEBUG_TRACKING_POINTER
void dump_(std::ostream &sout) const;
#endif

references_type refs_;
dependents_type deps_;
shared_ptr<Derived> self_;
boost::detail::atomic_count cnt_;
};

template<typename Derived>
inline void intrusive_ptr_add_ref(enable_reference_tracking<Derived> *p)
{
p->add_ref();
}

template<typename Derived>
inline void intrusive_ptr_release(enable_reference_tracking<Derived> *p)
{
p->release();
}

#ifdef BOOST_XPRESSIVE_DEBUG_TRACKING_POINTER
template<typename Derived>
inline void enable_reference_tracking<Derived>::dump_(std::ostream &sout) const
{
shared_ptr<Derived> this_ = this->self_;
sout << "0x" << (void*)this << " cnt=" << this_.use_count()-1 << " refs={";
typename references_type::const_iterator cur1 = this->refs_.begin();
typename references_type::const_iterator end1 = this->refs_.end();
for(; cur1 != end1; ++cur1)
{
sout << "0x" << (void*)&**cur1 << ',';
}
sout << "} deps={";
typename dependents_type::const_iterator cur2 = this->deps_.begin();
typename dependents_type::const_iterator end2 = this->deps_.end();
for(; cur2 != end2; ++cur2)
{
shared_ptr<Derived> dep = cur2->lock();
if(dep.get())
{
sout << "0x" << (void*)&*dep << ',';
}
}
sout << '}';
}
#endif

template<typename Type>
struct tracking_ptr
{
BOOST_MPL_ASSERT((is_base_and_derived<enable_reference_tracking<Type>, Type>));
typedef Type element_type;

tracking_ptr()
: impl_()
{
}

tracking_ptr(tracking_ptr<element_type> const &that)
: impl_()
{
this->operator =(that);
}

tracking_ptr<element_type> &operator =(tracking_ptr<element_type> const &that)
{
if(this != &that)
{
if(that)
{
if(that.has_deps_() || this->has_deps_())
{
this->fork_(); 
this->impl_->tracking_copy(*that);
}
else
{
this->impl_ = that.impl_; 
}
}
else if(*this)
{
this->impl_->tracking_clear();
}
}
return *this;
}

void swap(tracking_ptr<element_type> &that) 
{
this->impl_.swap(that.impl_);
}

shared_ptr<element_type> const &get() const
{
if(intrusive_ptr<element_type> impl = this->fork_())
{
this->impl_->tracking_copy(*impl);
}
return this->impl_->self_;
}

#if defined(__SUNPRO_CC) && BOOST_WORKAROUND(__SUNPRO_CC, <= 0x530)

operator bool() const
{
return this->impl_;
}

#else

typedef intrusive_ptr<element_type> tracking_ptr::* unspecified_bool_type;

operator unspecified_bool_type() const
{
return this->impl_ ? &tracking_ptr::impl_ : 0;
}

#endif

bool operator !() const
{
return !this->impl_;
}

element_type const *operator ->() const
{
return get_pointer(this->impl_);
}

element_type const &operator *() const
{
return *this->impl_;
}

private:

intrusive_ptr<element_type> fork_() const
{
intrusive_ptr<element_type> impl;
if(!this->impl_ || 1 != this->impl_->use_count())
{
impl = this->impl_;
BOOST_ASSERT(!this->has_deps_());
shared_ptr<element_type> simpl(new element_type);
this->impl_ = get_pointer(simpl->self_ = simpl);
}
return impl;
}

bool has_deps_() const
{
return this->impl_ && this->impl_->has_deps_();
}

mutable intrusive_ptr<element_type> impl_;
};

}}} 

#endif
