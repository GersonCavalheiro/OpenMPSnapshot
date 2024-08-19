
#ifndef BOOST_ICL_MAPALGO_HPP_JOFA_080225
#define BOOST_ICL_MAPALGO_HPP_JOFA_080225

#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>
#include <boost/icl/detail/notate.hpp>
#include <boost/icl/detail/set_algo.hpp>

#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4127) 
#endif                        

namespace boost{namespace icl
{
namespace Map 
{

template <class ObjectT, class CoObjectT>
bool intersects(const ObjectT& left, const CoObjectT& right)
{
typedef typename CoObjectT::const_iterator co_iterator;
co_iterator right_common_lower_, right_common_upper_;
if(!Set::common_range(right_common_lower_, right_common_upper_, right, left))
return false;

co_iterator right_ = right_common_lower_;
while(right_ != right_common_upper_)
if(!(left.find(key_value<CoObjectT>(right_++))==left.end()))
return true;

return false;
}


template<class MapT>
typename MapT::const_iterator next_proton(typename MapT::const_iterator& iter_, const MapT& object)
{
while(   iter_ != object.end() 
&& (*iter_).second == identity_element<typename MapT::codomain_type>::value())
++iter_;

return iter_;
}


template<class MapT>
bool lexicographical_distinct_equal(const MapT& left, const MapT& right)
{
if(&left == &right)        
return true;

typename MapT::const_iterator left_  = left.begin();
typename MapT::const_iterator right_ = right.begin();

left_  = next_proton(left_,  left);
right_ = next_proton(right_, right);

while(left_ != left.end() && right_ != right.end())
{
if(!(left_->first == right_->first && left_->second == right_->second))
return false;

++left_;
++right_;
left_  = next_proton(left_,  left);
right_ = next_proton(right_, right);
}

return left_ == left.end() && right_ == right.end();
}

} 
}} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

