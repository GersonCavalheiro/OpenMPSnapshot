
#ifndef BOOST_ICL_SET_ALGO_HPP_JOFA_990225
#define BOOST_ICL_SET_ALGO_HPP_JOFA_990225

#include <boost/type_traits/remove_const.hpp>
#include <boost/icl/detail/notate.hpp>
#include <boost/icl/concept/container.hpp>
#include <boost/icl/concept/set_value.hpp>
#include <boost/icl/concept/map_value.hpp>


namespace boost{namespace icl
{

namespace Set
{

template<class ObjectT, class ConstObjectT, class IteratorT>
bool common_range(IteratorT& lwb, IteratorT& upb, ObjectT& x1, const ConstObjectT& x2)
{
typedef typename ConstObjectT::const_iterator ConstObject_iterator;
typedef typename remove_const<ObjectT>::type  PureObjectT;

lwb = x1.end();
upb = x1.end();

if(icl::is_empty(x1) || icl::is_empty(x2)) 
return false;

IteratorT x1_fst_ = x1.begin();
IteratorT x1_lst_ = x1.end(); x1_lst_--;

ConstObject_iterator x2_fst_ = x2.begin();
ConstObject_iterator x2_lst_ = x2.end(); x2_lst_--;

typename ObjectT::key_compare key_less;
if(key_less(icl::key_value< PureObjectT>(x1_lst_), 
icl::key_value<ConstObjectT>(x2_fst_))) 
return false;
if(key_less(icl::key_value<ConstObjectT>(x2_lst_), 
icl::key_value< PureObjectT>(x1_fst_))) 
return false;

lwb = x1.lower_bound(icl::key_value<ConstObjectT>(x2_fst_));
upb = x1.upper_bound(icl::key_value<ConstObjectT>(x2_lst_));

return true;
}



template<class SetType>
inline bool within(const SetType& sub, const SetType& super)
{
if(&super == &sub)                   return true;
if(icl::is_empty(sub))               return true;
if(icl::is_empty(super))             return false;

typename SetType::const_iterator common_lwb_, common_upb_;
if(!common_range(common_lwb_, common_upb_, sub, super))
return false;

typename SetType::const_iterator sub_ = common_lwb_, super_;
while(sub_ != common_upb_)
{
super_ = super.find(*sub_++);
if(super_ == super.end()) 
return false;
}
return true;
}

template<class SetType>
bool intersects(const SetType& left, const SetType& right)
{
typename SetType::const_iterator common_lwb_right_, common_upb_right_;
if(!common_range(common_lwb_right_, common_upb_right_, right, left))
return false;

typename SetType::const_iterator right_ = common_lwb_right_, found_;
while(right_ != common_upb_right_)
{
found_ = left.find(*right_++);
if(found_ != left.end()) 
return true; 
}
return false;    
}


#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4996) 
#endif                        


template<class SetType>
inline bool lexicographical_equal(const SetType& left, const SetType& right)
{
if(&left == &right)
return true;
else return left.iterative_size() == right.iterative_size()
&& std::equal(left.begin(), left.end(), right.begin()); 
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


} 

}} 

#endif

