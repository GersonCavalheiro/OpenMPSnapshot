
#ifndef BOOST_ICL_CONCEPT_INTERVAL_MAP_HPP_JOFA_100920
#define BOOST_ICL_CONCEPT_INTERVAL_MAP_HPP_JOFA_100920

#include <boost/icl/type_traits/element_type_of.hpp>
#include <boost/icl/type_traits/segment_type_of.hpp>
#include <boost/icl/type_traits/absorbs_identities.hpp>
#include <boost/icl/type_traits/is_combinable.hpp>
#include <boost/icl/type_traits/is_interval_splitter.hpp>

#include <boost/icl/detail/set_algo.hpp>
#include <boost/icl/detail/interval_map_algo.hpp>
#include <boost/icl/concept/interval.hpp>
#include <boost/icl/concept/joinable.hpp>

namespace boost{ namespace icl
{

template<class Type>
inline typename enable_if<is_interval_map<Type>, typename Type::segment_type>::type
make_segment(const typename Type::element_type& element)
{
typedef typename Type::interval_type interval_type;
typedef typename Type::segment_type  segment_type;
return segment_type(icl::singleton<interval_type>(element.key), element.data);
}


template<class Type>
typename enable_if<is_interval_map<Type>, bool>::type
contains(const Type& super, const typename Type::element_type& key_value_pair)
{
typedef typename Type::const_iterator const_iterator;
const_iterator it_ = icl::find(super, key_value_pair.key);
return it_ != super.end() && (*it_).second == key_value_pair.data;
}

template<class Type>
typename enable_if<is_interval_map<Type>, bool>::type
contains(const Type& super, const typename Type::segment_type& sub_segment)
{
typedef typename Type::interval_type  interval_type;
typedef typename Type::const_iterator const_iterator;

interval_type sub_interval = sub_segment.first;
if(icl::is_empty(sub_interval)) 
return true;

std::pair<const_iterator, const_iterator> exterior = super.equal_range(sub_interval);
if(exterior.first == exterior.second)
return false;

const_iterator last_overlap = prior(exterior.second);

if(!(sub_segment.second == exterior.first->second) )
return false;

return
icl::contains(hull(exterior.first->first, last_overlap->first), sub_interval)
&&  Interval_Map::is_joinable(super, exterior.first, last_overlap);
}

template<class Type, class CoType>
typename enable_if<is_concept_compatible<is_interval_map, Type, CoType>, bool>::type
contains(const Type& super, const CoType& sub)
{
return Interval_Set::within(sub, super);
}


template<class Type, class CoType>
typename enable_if< mpl::and_< is_interval_map<Type>
, is_total<Type> 
, is_cross_derivative<Type, CoType> >
, bool>::type
contains(const Type&, const CoType&)
{
return true;
}

template<class Type>
typename enable_if< mpl::and_< is_interval_map<Type>
, mpl::not_<is_total<Type> > >
, bool>::type
contains(const Type& super, const typename Type::domain_type& key)    
{
return icl::find(super, key) != super.end();
}

template<class Type>
typename enable_if< mpl::and_< is_interval_map<Type>
, mpl::not_<is_total<Type> > >
, bool>::type
contains(const Type& super, const typename Type::interval_type& sub_interval)
{
typedef typename Type::const_iterator const_iterator;

if(icl::is_empty(sub_interval)) 
return true;

std::pair<const_iterator, const_iterator> exterior = super.equal_range(sub_interval);
if(exterior.first == exterior.second)
return false;

const_iterator last_overlap = prior(exterior.second);

return
icl::contains(hull(exterior.first->first, last_overlap->first), sub_interval)
&&  Interval_Set::is_joinable(super, exterior.first, last_overlap);
}

template<class Type, class KeyT>
typename enable_if< mpl::and_< is_concept_combinable<is_interval_map, is_interval_set, Type, KeyT>
, mpl::not_<is_total<Type> > >
, bool>::type
contains(const Type& super, const KeyT& sub)
{
return Interval_Set::within(sub, super);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
add(Type& object, const typename Type::segment_type& operand)
{
return object.add(operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
add(Type& object, const typename Type::element_type& operand)
{
return icl::add(object, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, typename Type::iterator >::type
add(Type& object, typename Type::iterator      prior_,
const typename Type::segment_type& operand)
{
return object.add(prior_, operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
insert(Type& object, const typename Type::segment_type& operand)
{
return object.insert(operand);
}

template<class Type>
inline typename enable_if<is_interval_map<Type>, Type>::type&
insert(Type& object, const typename Type::element_type& operand)
{
return icl::insert(object, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, typename Type::iterator>::type
insert(Type& object, typename Type::iterator      prior,
const typename Type::segment_type& operand)
{
return object.insert(prior, operand);
}


template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
erase(Type& object, const typename Type::interval_type& operand)
{
return object.erase(operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
erase(Type& object, const typename Type::domain_type& operand)
{
typedef typename Type::interval_type interval_type;
return icl::erase(object, icl::singleton<interval_type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
erase(Type& object, const typename Type::segment_type& operand)
{
return object.erase(operand);
}

template<class Type>
inline typename enable_if<is_interval_map<Type>, Type>::type&
erase(Type& object, const typename Type::element_type& operand)
{
return icl::erase(object, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type& 
subtract(Type& object, const typename Type::segment_type& operand)
{
return object.subtract(operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
subtract(Type& object, const typename Type::element_type& operand)
{
return icl::subtract(object, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
subtract(Type& object, const typename Type::domain_type& operand)
{
return object.erase(operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
subtract(Type& object, const typename Type::interval_type& operand)
{
return object.erase(operand);
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
set_at(Type& object, const typename Type::segment_type& operand)
{
icl::erase(object, operand.first); 
return icl::insert(object, operand); 
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
set_at(Type& object, const typename Type::element_type& operand)
{
return icl::set_at(object, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, void>::type
add_intersection(Type& section, const Type& object, 
const typename Type::element_type& operand)
{
object.add_intersection(section, make_segment<Type>(operand));
}

template<class Type>
typename enable_if<is_interval_map<Type>, void>::type
add_intersection(Type& section, const Type& object, 
const typename Type::segment_type& operand)
{
object.add_intersection(section, operand);
}

template<class Type, class MapT>
typename enable_if
<
mpl::and_< is_total<Type>
, is_concept_compatible<is_interval_map, Type, MapT> >
, void
>::type
add_intersection(Type& section, const Type& object, const MapT& operand)
{
section += object;
section += operand;
}

template<class Type, class MapT>
typename enable_if
<
mpl::and_< mpl::not_<is_total<Type> >
, is_concept_compatible<is_interval_map, Type, MapT> >
, void
>::type
add_intersection(Type& section, const Type& object, const MapT& operand)
{
typedef typename MapT::const_iterator const_iterator;

if(operand.empty()) 
return;
const_iterator common_lwb, common_upb;
if(!Set::common_range(common_lwb, common_upb, operand, object))
return;
const_iterator it_ = common_lwb;
while(it_ != common_upb)
add_intersection(section, object, *it_++);
}

template<class Type>
typename enable_if<is_interval_map<Type>, void>::type
add_intersection(Type& section, const Type& object, 
const typename Type::domain_type& key_value)
{
typedef typename Type::interval_type  interval_type;
typedef typename Type::segment_type   segment_type;
typedef typename Type::const_iterator const_iterator;

const_iterator it_ = icl::find(object, key_value);
if(it_ != object.end())
add(section, segment_type(interval_type(key_value),(*it_).second));
}

template<class Type>
typename enable_if<is_interval_map<Type>, void>::type
add_intersection(Type& section, const Type& object, 
const typename Type::interval_type& inter_val)
{
typedef typename Type::interval_type  interval_type;
typedef typename Type::value_type     value_type;
typedef typename Type::const_iterator const_iterator;
typedef typename Type::iterator       iterator;

if(icl::is_empty(inter_val)) 
return;

std::pair<const_iterator, const_iterator> exterior 
= object.equal_range(inter_val);
if(exterior.first == exterior.second)
return;

iterator prior_ = section.end();
for(const_iterator it_=exterior.first; it_ != exterior.second; it_++) 
{
interval_type common_interval = (*it_).first & inter_val; 
if(!icl::is_empty(common_interval))
prior_ = add(section, prior_, 
value_type(common_interval, (*it_).second) );
}
}

template<class Type, class KeySetT>
typename enable_if<is_concept_combinable<is_interval_map, is_interval_set, Type, KeySetT>, void>::type
add_intersection(Type& section, const Type& object, const KeySetT& key_set)
{
typedef typename KeySetT::const_iterator const_iterator;

if(icl::is_empty(key_set)) 
return;

const_iterator common_lwb, common_upb;
if(!Set::common_range(common_lwb, common_upb, key_set, object))
return;

const_iterator it_ = common_lwb;
while(it_ != common_upb)
add_intersection(section, object, *it_++);
}

template<class Type, class OperandT>
typename enable_if<mpl::and_< is_interval_map<Type>
, is_total<Type>
, boost::is_same< OperandT
, typename segment_type_of<Type>::type> >, 
bool>::type
intersects(const Type&, const OperandT&)
{
return true;
}

template<class Type, class OperandT>
typename enable_if<mpl::and_< is_interval_map<Type>
, mpl::not_<is_total<Type> >
, boost::is_same<OperandT, typename segment_type_of<Type>::type> >, 
bool>::type
intersects(const Type& object, const OperandT& operand)
{
Type intersection;
icl::add_intersection(intersection, object, operand);
return !icl::is_empty(intersection); 
}

template<class Type, class OperandT>
typename enable_if<mpl::and_< is_interval_map<Type>
, boost::is_same<OperandT, typename element_type_of<Type>::type> >, 
bool>::type
intersects(const Type& object, const OperandT& operand)
{
return icl::intersects(object, make_segment<Type>(operand)); 
}

template<class Type>
typename enable_if<is_interval_map<Type>, Type>::type&
flip(Type& object, const typename Type::segment_type& operand)
{
return object.flip(operand);
}

template<class Type>
inline typename enable_if<is_interval_map<Type>, Type>::type&
flip(Type& object, const typename Type::element_type& operand)
{
return icl::flip(object, make_segment<Type>(operand));
}

template<class Type, class OperandT>
typename enable_if< mpl::and_< is_total<Type>
, absorbs_identities<Type>
, is_concept_compatible<is_interval_map, 
Type, OperandT >
>
, Type>::type&
flip(Type& object, const OperandT&)
{
object.clear();
return object;
}

#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4127) 
#endif                        
template<class Type, class OperandT>
typename enable_if< mpl::and_< is_total<Type>
, mpl::not_<absorbs_identities<Type> >
, is_concept_compatible<is_interval_map, 
Type, OperandT >
>
, Type>::type&
flip(Type& object, const OperandT& operand)
{
typedef typename Type::codomain_type  codomain_type;

object += operand;
ICL_FORALL(typename Type, it_, object)
(*it_).second = identity_element<codomain_type>::value();

if(mpl::not_<is_interval_splitter<Type> >::value)
icl::join(object);

return object;
}
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


template<class Type, class OperandT>
typename enable_if< mpl::and_< mpl::not_<is_total<Type> > 
, is_concept_compatible<is_interval_map, 
Type, OperandT >
>
, Type>::type&
flip(Type& object, const OperandT& operand)
{
typedef typename OperandT::const_iterator const_iterator;

const_iterator common_lwb, common_upb;

if(!Set::common_range(common_lwb, common_upb, operand, object))
return object += operand;

const_iterator it_ = operand.begin();

while(it_ != common_lwb)
icl::add(object, *it_++);
while(it_ != common_upb)
icl::flip(object, *it_++);
while(it_ != operand.end())
icl::add(object, *it_++);

return object;
}

template<class Type, class SetT>
typename enable_if<is_concept_combinable<is_interval_set, is_interval_map, 
SetT, Type>, SetT>::type&
domain(SetT& result, const Type& object)
{
typedef typename SetT::iterator set_iterator;
result.clear(); 
set_iterator prior_ = result.end();
ICL_const_FORALL(typename Type, it_, object) 
prior_ = icl::insert(result, prior_, (*it_).first); 

return result;
}

template<class Type, class SetT>
typename enable_if<is_concept_combinable<is_interval_set, is_interval_map, 
SetT, Type>, SetT>::type&
between(SetT& in_between, const Type& object)
{
typedef typename Type::const_iterator const_iterator;
typedef typename SetT::iterator       set_iterator;
in_between.clear();
const_iterator it_ = object.begin(), pred_;
set_iterator   prior_ = in_between.end();

if(it_ != object.end())
pred_ = it_++;

while(it_ != object.end())
prior_ = icl::insert(in_between, prior_, 
between((*pred_++).first, (*it_++).first));

return in_between;
}

template<class MapT, class Predicate>
typename enable_if<is_interval_map<MapT>, MapT>::type&
erase_if(const Predicate& pred, MapT& object)
{
typename MapT::iterator it_ = object.begin();
while(it_ != object.end())
if(pred(*it_))
object.erase(it_++); 
else ++it_;
return object;
}

template<class MapT, class Predicate>
inline typename enable_if<is_interval_map<MapT>, MapT>::type&
add_if(const Predicate& pred, MapT& object, const MapT& src)
{
typename MapT::const_iterator it_ = src.begin();
while(it_ != src.end())
if(pred(*it_)) 
icl::add(object, *it_++); 

return object;
}

template<class MapT, class Predicate>
inline typename enable_if<is_interval_map<MapT>, MapT>::type&
assign_if(const Predicate& pred, MapT& object, const MapT& src)
{
icl::clear(object);
return add_if(object, src, pred);
}


template<class Type>
typename enable_if<mpl::and_< is_interval_map<Type>
, absorbs_identities<Type> >, Type>::type&
absorb_identities(Type& object)
{
return object;
}

template<class Type>
typename enable_if<mpl::and_< is_interval_map<Type>
, mpl::not_<absorbs_identities<Type> > >, Type>::type&
absorb_identities(Type& object)
{
typedef typename Type::segment_type segment_type;
return icl::erase_if(content_is_identity_element<segment_type>(), object);
}

template<class CharType, class CharTraits, class Type>
typename enable_if<is_interval_map<Type>, 
std::basic_ostream<CharType, CharTraits> >::type& 
operator << (std::basic_ostream<CharType, CharTraits>& stream, const Type& object)
{
stream << "{";
ICL_const_FORALL(typename Type, it_, object)
stream << "(" << (*it_).first << "->" << (*it_).second << ")";

return stream << "}";
}


}} 

#endif


