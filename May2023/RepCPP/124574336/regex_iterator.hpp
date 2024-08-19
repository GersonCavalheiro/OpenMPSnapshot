
#ifndef BOOST_XPRESSIVE_REGEX_ITERATOR_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_REGEX_ITERATOR_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/access.hpp>
#include <boost/xpressive/detail/utility/counted_base.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename BidiIter>
struct regex_iterator_impl
: counted_base<regex_iterator_impl<BidiIter> >
{
typedef detail::core_access<BidiIter> access;

regex_iterator_impl
(
BidiIter begin
, BidiIter cur
, BidiIter end
, BidiIter next_search
, basic_regex<BidiIter> const &rex
, regex_constants::match_flag_type flags
, bool not_null = false
)
: rex_(rex)
, what_()
, state_(begin, end, what_, *access::get_regex_impl(rex_), flags)
, flags_(flags)
, not_null_(not_null)
{
this->state_.cur_ = cur;
this->state_.next_search_ = next_search;
}

bool next()
{
this->state_.reset(this->what_, *access::get_regex_impl(this->rex_));
if(!regex_search_impl(this->state_, this->rex_, this->not_null_))
{
return false;
}

access::set_base(this->what_, this->state_.begin_);

this->state_.cur_ = this->state_.next_search_ = this->what_[0].second;
this->not_null_ = (0 == this->what_.length());

return true;
}

bool equal_to(regex_iterator_impl<BidiIter> const &that) const
{
return this->rex_.regex_id()    == that.rex_.regex_id()
&& this->state_.begin_      == that.state_.begin_
&& this->state_.cur_        == that.state_.cur_
&& this->state_.end_        == that.state_.end_
&& this->flags_             == that.flags_
;
}

basic_regex<BidiIter> rex_;
match_results<BidiIter> what_;
match_state<BidiIter> state_;
regex_constants::match_flag_type const flags_;
bool not_null_;
};

} 

template<typename BidiIter>
struct regex_iterator
{
typedef basic_regex<BidiIter> regex_type;
typedef match_results<BidiIter> value_type;
typedef typename iterator_difference<BidiIter>::type difference_type;
typedef value_type const *pointer;
typedef value_type const &reference;
typedef std::forward_iterator_tag iterator_category;

typedef detail::regex_iterator_impl<BidiIter> impl_type_;

regex_iterator()
: impl_()
{
}

regex_iterator
(
BidiIter begin
, BidiIter end
, basic_regex<BidiIter> const &rex
, regex_constants::match_flag_type flags = regex_constants::match_default
)
: impl_()
{
if(0 != rex.regex_id()) 
{
this->impl_ = new impl_type_(begin, begin, end, begin, rex, flags);
this->next_();
}
}

template<typename LetExpr>
regex_iterator
(
BidiIter begin
, BidiIter end
, basic_regex<BidiIter> const &rex
, detail::let_<LetExpr> const &args
, regex_constants::match_flag_type flags = regex_constants::match_default
)
: impl_()
{
if(0 != rex.regex_id()) 
{
this->impl_ = new impl_type_(begin, begin, end, begin, rex, flags);
detail::bind_args(args, this->impl_->what_);
this->next_();
}
}

regex_iterator(regex_iterator<BidiIter> const &that)
: impl_(that.impl_) 
{
}

regex_iterator<BidiIter> &operator =(regex_iterator<BidiIter> const &that)
{
this->impl_ = that.impl_; 
return *this;
}

friend bool operator ==(regex_iterator<BidiIter> const &left, regex_iterator<BidiIter> const &right)
{
if(!left.impl_ || !right.impl_)
{
return !left.impl_ && !right.impl_;
}

return left.impl_->equal_to(*right.impl_);
}

friend bool operator !=(regex_iterator<BidiIter> const &left, regex_iterator<BidiIter> const &right)
{
return !(left == right);
}

value_type const &operator *() const
{
return this->impl_->what_;
}

value_type const *operator ->() const
{
return &this->impl_->what_;
}

regex_iterator<BidiIter> &operator ++()
{
this->fork_(); 
this->next_();
return *this;
}

regex_iterator<BidiIter> operator ++(int)
{
regex_iterator<BidiIter> tmp(*this);
++*this;
return tmp;
}

private:

void fork_()
{
if(1 != this->impl_->use_count())
{
impl_type_ *that = this->impl_.get();
this->impl_ = new impl_type_
(
that->state_.begin_
, that->state_.cur_
, that->state_.end_
, that->state_.next_search_
, that->rex_
, that->flags_
, that->not_null_
);
detail::core_access<BidiIter>::get_action_args(this->impl_->what_)
= detail::core_access<BidiIter>::get_action_args(that->what_);
}
}

void next_()
{
BOOST_ASSERT(this->impl_ && 1 == this->impl_->use_count());
if(!this->impl_->next())
{
this->impl_ = 0;
}
}

intrusive_ptr<impl_type_> impl_;
};

}} 

#endif
