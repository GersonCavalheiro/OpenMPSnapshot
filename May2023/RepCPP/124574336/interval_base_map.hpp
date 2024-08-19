
#ifndef BOOST_ICL_INTERVAL_BASE_MAP_HPP_JOFA_990223
#define BOOST_ICL_INTERVAL_BASE_MAP_HPP_JOFA_990223

#include <limits>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>

#include <boost/icl/detail/notate.hpp> 
#include <boost/icl/detail/design_config.hpp>
#include <boost/icl/detail/on_absorbtion.hpp>
#include <boost/icl/detail/interval_map_algo.hpp>
#include <boost/icl/detail/exclusive_less_than.hpp>

#include <boost/icl/associative_interval_container.hpp>

#include <boost/icl/type_traits/is_interval_splitter.hpp>
#include <boost/icl/map.hpp>

namespace boost{namespace icl
{

template<class DomainT, class CodomainT>
struct mapping_pair
{
DomainT   key;
CodomainT data;

mapping_pair():key(), data(){}

mapping_pair(const DomainT& key_value, const CodomainT& data_value)
:key(key_value), data(data_value){}

mapping_pair(const std::pair<DomainT,CodomainT>& std_pair)
:key(std_pair.first), data(std_pair.second){}
};


template
<
class SubType,
typename DomainT,
typename CodomainT,
class Traits = icl::partial_absorber,
ICL_COMPARE Compare  = ICL_COMPARE_INSTANCE(ICL_COMPARE_DEFAULT, DomainT),
ICL_COMBINE Combine  = ICL_COMBINE_INSTANCE(icl::inplace_plus, CodomainT),
ICL_SECTION Section  = ICL_SECTION_INSTANCE(icl::inter_section, CodomainT), 
ICL_INTERVAL(ICL_COMPARE) Interval = ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, DomainT, Compare),
ICL_ALLOC   Alloc    = std::allocator
>
class interval_base_map
{
public:
typedef interval_base_map<SubType,DomainT,CodomainT,
Traits,Compare,Combine,Section,Interval,Alloc>
type;

typedef SubType sub_type;

typedef type overloadable_type;

typedef Traits traits;

typedef typename icl::map<DomainT,CodomainT,
Traits,Compare,Combine,Section,Alloc> atomized_type;

typedef DomainT   domain_type;
typedef typename boost::call_traits<DomainT>::param_type domain_param;
typedef CodomainT codomain_type;
typedef mapping_pair<domain_type,codomain_type> domain_mapping_type;
typedef domain_mapping_type element_type;
typedef ICL_INTERVAL_TYPE(Interval,DomainT,Compare) interval_type;
typedef std::pair<interval_type,CodomainT> interval_mapping_type;
typedef std::pair<interval_type,CodomainT> segment_type;

typedef typename difference_type_of<domain_type>::type difference_type;
typedef typename size_type_of<domain_type>::type size_type;

typedef ICL_COMPARE_DOMAIN(Compare,DomainT)      domain_compare;
typedef ICL_COMPARE_DOMAIN(Compare,segment_type) segment_compare;
typedef ICL_COMBINE_CODOMAIN(Combine,CodomainT)  codomain_combine;
typedef typename inverse<codomain_combine>::type inverse_codomain_combine;

typedef typename mpl::if_
<has_set_semantics<codomain_type>
, ICL_SECTION_CODOMAIN(Section,CodomainT)     
, codomain_combine
>::type                                            codomain_intersect;


typedef typename inverse<codomain_intersect>::type inverse_codomain_intersect;

typedef exclusive_less_than<interval_type> interval_compare;

typedef exclusive_less_than<interval_type> key_compare;

typedef Alloc<std::pair<const interval_type, codomain_type> > 
allocator_type;

typedef ICL_IMPL_SPACE::map<interval_type,codomain_type,
key_compare,allocator_type> ImplMapT;

typedef typename ImplMapT::key_type   key_type;
typedef typename ImplMapT::value_type value_type;
typedef typename ImplMapT::value_type::second_type data_type;

typedef typename ImplMapT::pointer         pointer;
typedef typename ImplMapT::const_pointer   const_pointer;
typedef typename ImplMapT::reference       reference;
typedef typename ImplMapT::const_reference const_reference;

typedef typename ImplMapT::iterator iterator;
typedef typename ImplMapT::const_iterator const_iterator;
typedef typename ImplMapT::reverse_iterator reverse_iterator;
typedef typename ImplMapT::const_reverse_iterator const_reverse_iterator;

typedef boost::icl::element_iterator<iterator> element_iterator; 
typedef boost::icl::element_iterator<const_iterator> element_const_iterator; 
typedef boost::icl::element_iterator<reverse_iterator> element_reverse_iterator; 
typedef boost::icl::element_iterator<const_reverse_iterator> element_const_reverse_iterator; 

typedef typename on_absorbtion<type, codomain_combine, 
Traits::absorbs_identities>::type on_codomain_absorbtion;

public:
BOOST_STATIC_CONSTANT(bool, 
is_total_invertible = (   Traits::is_total 
&& has_inverse<codomain_type>::value));

BOOST_STATIC_CONSTANT(int, fineness = 0); 

public:


interval_base_map()
{
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<DomainT>));
BOOST_CONCEPT_ASSERT((LessThanComparableConcept<DomainT>));
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<CodomainT>));
BOOST_CONCEPT_ASSERT((EqualComparableConcept<CodomainT>));
}


interval_base_map(const interval_base_map& src): _map(src._map)
{
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<DomainT>));
BOOST_CONCEPT_ASSERT((LessThanComparableConcept<DomainT>));
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<CodomainT>));
BOOST_CONCEPT_ASSERT((EqualComparableConcept<CodomainT>));
}

#   ifndef BOOST_ICL_NO_CXX11_RVALUE_REFERENCES


interval_base_map(interval_base_map&& src): _map(boost::move(src._map))
{
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<DomainT>));
BOOST_CONCEPT_ASSERT((LessThanComparableConcept<DomainT>));
BOOST_CONCEPT_ASSERT((DefaultConstructibleConcept<CodomainT>));
BOOST_CONCEPT_ASSERT((EqualComparableConcept<CodomainT>));
}


interval_base_map& operator = (interval_base_map src) 
{                           
this->_map = boost::move(src._map);
return *this; 
}

#   else 


interval_base_map& operator = (const interval_base_map& src) 
{ 
this->_map = src._map;
return *this; 
}

#   endif 


void swap(interval_base_map& object) { _map.swap(object._map); }


void clear() { icl::clear(*that()); }


bool empty()const { return icl::is_empty(*that()); }


size_type size()const
{
return icl::cardinality(*that());
}


std::size_t iterative_size()const 
{ 
return _map.size(); 
}



const_iterator find(const domain_type& key_value)const
{ 
return icl::find(*this, key_value);
}


const_iterator find(const interval_type& key_interval)const
{ 
return _map.find(key_interval); 
}


codomain_type operator()(const domain_type& key_value)const
{
const_iterator it_ = icl::find(*this, key_value);
return it_==end() ? identity_element<codomain_type>::value()
: (*it_).second;
}



SubType& add(const element_type& key_value_pair) 
{
return icl::add(*that(), key_value_pair);
}


SubType& add(const segment_type& interval_value_pair) 
{
this->template _add<codomain_combine>(interval_value_pair);
return *that();
}


iterator add(iterator prior_, const segment_type& interval_value_pair) 
{
return this->template _add<codomain_combine>(prior_, interval_value_pair);
}


SubType& subtract(const element_type& key_value_pair)
{ 
return icl::subtract(*that(), key_value_pair);
}


SubType& subtract(const segment_type& interval_value_pair)
{
on_invertible<type, is_total_invertible>
::subtract(*that(), interval_value_pair);
return *that();
}


SubType& insert(const element_type& key_value_pair) 
{
return icl::insert(*that(), key_value_pair);
}


SubType& insert(const segment_type& interval_value_pair)
{ 
_insert(interval_value_pair); 
return *that();
}


iterator insert(iterator prior, const segment_type& interval_value_pair)
{ 
return _insert(prior, interval_value_pair);
}


SubType& set(const element_type& key_value_pair) 
{ 
return icl::set_at(*that(), key_value_pair);
}


SubType& set(const segment_type& interval_value_pair)
{ 
return icl::set_at(*that(), interval_value_pair);
}


SubType& erase(const element_type& key_value_pair) 
{ 
icl::erase(*that(), key_value_pair);
return *that();
}


SubType& erase(const segment_type& interval_value_pair);


SubType& erase(const domain_type& key) 
{ 
return icl::erase(*that(), key); 
}


SubType& erase(const interval_type& inter_val);



void erase(iterator position){ this->_map.erase(position); }


void erase(iterator first, iterator past){ this->_map.erase(first, past); }


void add_intersection(SubType& section, const segment_type& interval_value_pair)const
{
on_definedness<SubType, Traits::is_total>
::add_intersection(section, *that(), interval_value_pair);
}


SubType& flip(const element_type& key_value_pair)
{ 
return icl::flip(*that(), key_value_pair); 
}


SubType& flip(const segment_type& interval_value_pair)
{
on_total_absorbable<SubType, Traits::is_total, Traits::absorbs_identities>
::flip(*that(), interval_value_pair);
return *that();
}


iterator lower_bound(const key_type& interval)
{ return _map.lower_bound(interval); }

iterator upper_bound(const key_type& interval)
{ return _map.upper_bound(interval); }

const_iterator lower_bound(const key_type& interval)const
{ return _map.lower_bound(interval); }

const_iterator upper_bound(const key_type& interval)const
{ return _map.upper_bound(interval); }

std::pair<iterator,iterator> equal_range(const key_type& interval)
{ 
return std::pair<iterator,iterator>
(lower_bound(interval), upper_bound(interval)); 
}

std::pair<const_iterator,const_iterator> 
equal_range(const key_type& interval)const
{ 
return std::pair<const_iterator,const_iterator>
(lower_bound(interval), upper_bound(interval)); 
}

iterator begin() { return _map.begin(); }
iterator end()   { return _map.end(); }
const_iterator begin()const { return _map.begin(); }
const_iterator end()const   { return _map.end(); }
reverse_iterator rbegin() { return _map.rbegin(); }
reverse_iterator rend()   { return _map.rend(); }
const_reverse_iterator rbegin()const { return _map.rbegin(); }
const_reverse_iterator rend()const   { return _map.rend(); }

private:
template<class Combiner>
iterator _add(const segment_type& interval_value_pair);

template<class Combiner>
iterator _add(iterator prior_, const segment_type& interval_value_pair);

template<class Combiner>
void _subtract(const segment_type& interval_value_pair);

iterator _insert(const segment_type& interval_value_pair);
iterator _insert(iterator prior_, const segment_type& interval_value_pair);

private:
template<class Combiner>
void add_segment(const interval_type& inter_val, const CodomainT& co_val, iterator& it_);

template<class Combiner>
void add_main(interval_type& inter_val, const CodomainT& co_val, 
iterator& it_, const iterator& last_);

template<class Combiner>
void add_rear(const interval_type& inter_val, const CodomainT& co_val, iterator& it_);

void add_front(const interval_type& inter_val, iterator& first_);

private:
void subtract_front(const interval_type& inter_val, iterator& first_);

template<class Combiner>
void subtract_main(const CodomainT& co_val, iterator& it_, const iterator& last_);

template<class Combiner>
void subtract_rear(interval_type& inter_val, const CodomainT& co_val, iterator& it_);

private:
void insert_main(const interval_type&, const CodomainT&, iterator&, const iterator&);
void erase_rest (      interval_type&, const CodomainT&, iterator&, const iterator&);

template<class FragmentT>
void total_add_intersection(SubType& section, const FragmentT& fragment)const
{
section += *that();
section.add(fragment);
}

void partial_add_intersection(SubType& section, const segment_type& operand)const
{
interval_type inter_val = operand.first;
if(icl::is_empty(inter_val)) 
return;

std::pair<const_iterator, const_iterator> exterior = equal_range(inter_val);
if(exterior.first == exterior.second)
return;

for(const_iterator it_=exterior.first; it_ != exterior.second; it_++) 
{
interval_type common_interval = (*it_).first & inter_val; 
if(!icl::is_empty(common_interval))
{
section.template _add<codomain_combine>  (value_type(common_interval, (*it_).second) );
section.template _add<codomain_intersect>(value_type(common_interval, operand.second));
}
}
}

void partial_add_intersection(SubType& section, const element_type& operand)const
{
partial_add_intersection(section, make_segment<type>(operand));
}


protected:

template <class Combiner>
iterator gap_insert(iterator prior_, const interval_type& inter_val, 
const codomain_type& co_val   )
{
BOOST_ASSERT(this->_map.find(inter_val) == this->_map.end());
BOOST_ASSERT((!on_absorbtion<type,Combiner,Traits::absorbs_identities>::is_absorbable(co_val)));
return this->_map.insert(prior_, value_type(inter_val, version<Combiner>()(co_val)));
}

template <class Combiner>
std::pair<iterator, bool>
add_at(const iterator& prior_, const interval_type& inter_val, 
const codomain_type& co_val   )
{
BOOST_ASSERT((!(on_absorbtion<type,Combiner,Traits::absorbs_identities>::is_absorbable(co_val))));

iterator inserted_ 
= this->_map.insert(prior_, value_type(inter_val, Combiner::identity_element()));

if((*inserted_).first == inter_val && (*inserted_).second == Combiner::identity_element())
{
Combiner()((*inserted_).second, co_val);
return std::pair<iterator,bool>(inserted_, true);
}
else
return std::pair<iterator,bool>(inserted_, false);
}

std::pair<iterator, bool>
insert_at(const iterator& prior_, const interval_type& inter_val, 
const codomain_type& co_val   )
{
iterator inserted_
= this->_map.insert(prior_, value_type(inter_val, co_val));

if(inserted_ == prior_)
return std::pair<iterator,bool>(inserted_, false);
else if((*inserted_).first == inter_val)
return std::pair<iterator,bool>(inserted_, true);
else
return std::pair<iterator,bool>(inserted_, false);
}


protected:
sub_type* that() { return static_cast<sub_type*>(this); }
const sub_type* that()const { return static_cast<const sub_type*>(this); }

protected:
ImplMapT _map;


private:
template<class Type, bool is_total_invertible>
struct on_invertible;

template<class Type>
struct on_invertible<Type, true>
{
typedef typename Type::segment_type segment_type;
typedef typename Type::inverse_codomain_combine inverse_codomain_combine;

static void subtract(Type& object, const segment_type& operand)
{ object.template _add<inverse_codomain_combine>(operand); }
};

template<class Type>
struct on_invertible<Type, false>
{
typedef typename Type::segment_type segment_type;
typedef typename Type::inverse_codomain_combine inverse_codomain_combine;

static void subtract(Type& object, const segment_type& operand)
{ object.template _subtract<inverse_codomain_combine>(operand); }
};

friend struct on_invertible<type, true>;
friend struct on_invertible<type, false>;

template<class Type, bool is_total>
struct on_definedness;

template<class Type>
struct on_definedness<Type, true>
{
static void add_intersection(Type& section, const Type& object, 
const segment_type& operand)
{ object.total_add_intersection(section, operand); }
};

template<class Type>
struct on_definedness<Type, false>
{
static void add_intersection(Type& section, const Type& object, 
const segment_type& operand)
{ object.partial_add_intersection(section, operand); }
};

friend struct on_definedness<type, true>;
friend struct on_definedness<type, false>;

template<class Type, bool has_set_semantics> 
struct on_codomain_model;

template<class Type>
struct on_codomain_model<Type, true>
{
typedef typename Type::interval_type interval_type;
typedef typename Type::codomain_type codomain_type;
typedef typename Type::segment_type  segment_type;
typedef typename Type::codomain_combine codomain_combine;
typedef typename Type::inverse_codomain_intersect inverse_codomain_intersect;

static void add(Type& intersection, interval_type& common_interval, 
const codomain_type& flip_value, const codomain_type& co_value)
{
codomain_type common_value = flip_value;
inverse_codomain_intersect()(common_value, co_value);
intersection.template 
_add<codomain_combine>(segment_type(common_interval, common_value));
}
};

template<class Type>
struct on_codomain_model<Type, false>
{
typedef typename Type::interval_type interval_type;
typedef typename Type::codomain_type codomain_type;
typedef typename Type::segment_type  segment_type;
typedef typename Type::codomain_combine codomain_combine;

static void add(Type& intersection, interval_type& common_interval, 
const codomain_type&, const codomain_type&)
{
intersection.template 
_add<codomain_combine>(segment_type(common_interval, 
identity_element<codomain_type>::value()));
}
};

friend struct on_codomain_model<type, true>;
friend struct on_codomain_model<type, false>;


template<class Type, bool is_total, bool absorbs_identities>
struct on_total_absorbable;

template<class Type>
struct on_total_absorbable<Type, true, true>
{
static void flip(Type& object, const typename Type::segment_type&)
{ icl::clear(object); }
};

#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4127) 
#endif                        

template<class Type>
struct on_total_absorbable<Type, true, false>
{
typedef typename Type::segment_type  segment_type;
typedef typename Type::codomain_type codomain_type;

static void flip(Type& object, const segment_type& operand)
{ 
object += operand;
ICL_FORALL(typename Type, it_, object)
(*it_).second = identity_element<codomain_type>::value();

if(mpl::not_<is_interval_splitter<Type> >::value)
icl::join(object);
}
};

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template<class Type, bool absorbs_identities>
struct on_total_absorbable<Type, false, absorbs_identities>
{
typedef typename Type::segment_type   segment_type;
typedef typename Type::codomain_type  codomain_type;
typedef typename Type::interval_type  interval_type;
typedef typename Type::value_type     value_type;
typedef typename Type::const_iterator const_iterator;
typedef typename Type::set_type       set_type;
typedef typename Type::inverse_codomain_intersect inverse_codomain_intersect;

static void flip(Type& object, const segment_type& interval_value_pair)
{
interval_type span = interval_value_pair.first;
std::pair<const_iterator, const_iterator> exterior 
= object.equal_range(span);

const_iterator first_ = exterior.first;
const_iterator end_   = exterior.second;

interval_type covered, left_over, common_interval;
const codomain_type& x_value = interval_value_pair.second;
const_iterator it_ = first_;

set_type eraser;
Type     intersection;

while(it_ != end_  ) 
{
const codomain_type& co_value = (*it_).second;
covered = (*it_++).first;
left_over = right_subtract(span, covered);

common_interval = span & covered;
if(!icl::is_empty(common_interval))
{
icl::add(eraser, common_interval);

on_codomain_model<Type, has_set_semantics<codomain_type>::value>
::add(intersection, common_interval, x_value, co_value);
}

icl::add(object, value_type(left_over, x_value)); 

span = left_subtract(span, covered);
}

icl::add(object, value_type(span, x_value));

icl::erase(object, eraser);
object += intersection;
}
};
} ;


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::add_front(const interval_type& inter_val, iterator& first_)
{

interval_type left_resid = right_subtract((*first_).first, inter_val);

if(!icl::is_empty(left_resid))
{   
iterator prior_ = cyclic_prior(*this, first_);
const_cast<interval_type&>((*first_).first) 
= left_subtract((*first_).first, left_resid);
this->_map.insert(prior_, segment_type(left_resid, (*first_).second));
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::add_segment(const interval_type& inter_val, const CodomainT& co_val, iterator& it_)
{
interval_type lead_gap = right_subtract(inter_val, (*it_).first);
if(!icl::is_empty(lead_gap))
{
iterator prior_ = it_==this->_map.begin()? it_ : prior(it_); 
iterator inserted_ = this->template gap_insert<Combiner>(prior_, lead_gap, co_val);
that()->handle_inserted(prior_, inserted_);
}

Combiner()((*it_).second, co_val);
that()->template handle_left_combined<Combiner>(it_++);
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::add_main(interval_type& inter_val, const CodomainT& co_val, 
iterator& it_, const iterator& last_)
{
interval_type cur_interval;
while(it_!=last_)
{
cur_interval = (*it_).first ;
add_segment<Combiner>(inter_val, co_val, it_);
inter_val = left_subtract(inter_val, cur_interval);
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::add_rear(const interval_type& inter_val, const CodomainT& co_val, iterator& it_)
{
iterator prior_ = cyclic_prior(*that(), it_);
interval_type cur_itv = (*it_).first ;

interval_type lead_gap = right_subtract(inter_val, cur_itv);
if(!icl::is_empty(lead_gap))
{   
iterator inserted_ = this->template gap_insert<Combiner>(prior_, lead_gap, co_val);
that()->handle_inserted(prior_, inserted_);
}

interval_type end_gap = left_subtract(inter_val, cur_itv);
if(!icl::is_empty(end_gap))
{
Combiner()((*it_).second, co_val);
that()->template gap_insert_at<Combiner>(it_, prior_, end_gap, co_val);
}
else
{
interval_type right_resid = left_subtract(cur_itv, inter_val);

if(icl::is_empty(right_resid))
{
Combiner()((*it_).second, co_val);
that()->template handle_preceeded_combined<Combiner>(prior_, it_);
}
else
{
const_cast<interval_type&>((*it_).first) = right_subtract((*it_).first, right_resid);

iterator insertion_ = this->_map.insert(it_, value_type(right_resid, (*it_).second));
that()->handle_reinserted(insertion_);

Combiner()((*it_).second, co_val);
that()->template handle_preceeded_combined<Combiner>(insertion_, it_);
}
}
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline typename interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>::iterator
interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::_add(const segment_type& addend)
{
typedef typename on_absorbtion<type,Combiner,
absorbs_identities<type>::value>::type on_absorbtion_;

const interval_type& inter_val = addend.first;
if(icl::is_empty(inter_val)) 
return this->_map.end();

const codomain_type& co_val = addend.second;
if(on_absorbtion_::is_absorbable(co_val))
return this->_map.end();

std::pair<iterator,bool> insertion 
= this->_map.insert(value_type(inter_val, version<Combiner>()(co_val)));

if(insertion.second)
return that()->handle_inserted(insertion.first);
else
{
iterator first_ = this->_map.lower_bound(inter_val),
last_  = prior(this->_map.upper_bound(inter_val));
iterator it_ = first_;
interval_type rest_interval = inter_val;

add_front         (rest_interval,         it_       );
add_main<Combiner>(rest_interval, co_val, it_, last_);
add_rear<Combiner>(rest_interval, co_val, it_       );
return it_;
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline typename interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>::iterator
interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::_add(iterator prior_, const segment_type& addend)
{
typedef typename on_absorbtion<type,Combiner,
absorbs_identities<type>::value>::type on_absorbtion_;

const interval_type& inter_val = addend.first;
if(icl::is_empty(inter_val)) 
return prior_;

const codomain_type& co_val = addend.second;
if(on_absorbtion_::is_absorbable(co_val))
return prior_;

std::pair<iterator,bool> insertion 
= add_at<Combiner>(prior_, inter_val, co_val);

if(insertion.second)
return that()->handle_inserted(insertion.first);
else
{
std::pair<iterator,iterator> overlap = equal_range(inter_val);
iterator it_   = overlap.first,
last_ = prior(overlap.second);
interval_type rest_interval = inter_val;

add_front         (rest_interval,         it_       );
add_main<Combiner>(rest_interval, co_val, it_, last_);
add_rear<Combiner>(rest_interval, co_val, it_       );
return it_;
}
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::subtract_front(const interval_type& inter_val, iterator& it_)
{
interval_type left_resid = right_subtract((*it_).first, inter_val);

if(!icl::is_empty(left_resid)) 
{                              
iterator prior_ = cyclic_prior(*this, it_); 
const_cast<interval_type&>((*it_).first) = left_subtract((*it_).first, left_resid);
this->_map.insert(prior_, value_type(left_resid, (*it_).second));
}
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::subtract_main(const CodomainT& co_val, iterator& it_, const iterator& last_)
{
while(it_ != last_)
{
Combiner()((*it_).second, co_val);
that()->template handle_left_combined<Combiner>(it_++);
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::subtract_rear(interval_type& inter_val, const CodomainT& co_val, iterator& it_)
{
interval_type right_resid = left_subtract((*it_).first, inter_val);

if(icl::is_empty(right_resid))
{
Combiner()((*it_).second, co_val);
that()->template handle_combined<Combiner>(it_);
}
else
{
const_cast<interval_type&>((*it_).first) = right_subtract((*it_).first, right_resid);
iterator next_ = this->_map.insert(it_, value_type(right_resid, (*it_).second));
Combiner()((*it_).second, co_val);
that()->template handle_succeeded_combined<Combiner>(it_, next_);
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
template<class Combiner>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::_subtract(const segment_type& minuend)
{
interval_type inter_val = minuend.first;
if(icl::is_empty(inter_val)) 
return;

const codomain_type& co_val = minuend.second;
if(on_absorbtion<type,Combiner,Traits::absorbs_identities>::is_absorbable(co_val)) 
return;

std::pair<iterator, iterator> exterior = equal_range(inter_val);
if(exterior.first == exterior.second)
return;

iterator last_  = prior(exterior.second);
iterator it_    = exterior.first;
subtract_front          (inter_val,         it_       );
subtract_main <Combiner>(           co_val, it_, last_);
subtract_rear <Combiner>(inter_val, co_val, it_       );
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::insert_main(const interval_type& inter_val, const CodomainT& co_val, 
iterator& it_, const iterator& last_)
{
iterator end_   = boost::next(last_);
iterator prior_ = cyclic_prior(*this,it_), inserted_;
interval_type rest_interval = inter_val, left_gap, cur_itv;
interval_type last_interval = last_ ->first;

while(it_ != end_  )
{
cur_itv = (*it_).first ;            
left_gap = right_subtract(rest_interval, cur_itv);

if(!icl::is_empty(left_gap))
{
inserted_ = this->_map.insert(prior_, value_type(left_gap, co_val));
it_ = that()->handle_inserted(inserted_);
}

rest_interval = left_subtract(rest_interval, cur_itv);
prior_ = it_;
++it_;
}

interval_type end_gap = left_subtract(rest_interval, last_interval);
if(!icl::is_empty(end_gap))
{
inserted_ = this->_map.insert(prior_, value_type(end_gap, co_val));
it_ = that()->handle_inserted(inserted_);
}
else
it_ = prior_;
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline typename interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>::iterator
interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::_insert(const segment_type& addend)
{
interval_type inter_val = addend.first;
if(icl::is_empty(inter_val)) 
return this->_map.end();

const codomain_type& co_val = addend.second;
if(on_codomain_absorbtion::is_absorbable(co_val)) 
return this->_map.end();

std::pair<iterator,bool> insertion = this->_map.insert(addend);

if(insertion.second)
return that()->handle_inserted(insertion.first);
else
{
iterator first_ = this->_map.lower_bound(inter_val),
last_  = prior(this->_map.upper_bound(inter_val));
iterator it_ = first_;
insert_main(inter_val, co_val, it_, last_);
return it_;
}
}


template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline typename interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>::iterator
interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::_insert(iterator prior_, const segment_type& addend)
{
interval_type inter_val = addend.first;
if(icl::is_empty(inter_val)) 
return prior_;

const codomain_type& co_val = addend.second;
if(on_codomain_absorbtion::is_absorbable(co_val)) 
return prior_;

std::pair<iterator,bool> insertion = insert_at(prior_, inter_val, co_val);

if(insertion.second)
return that()->handle_inserted(insertion.first);
{
std::pair<iterator,iterator> overlap = equal_range(inter_val);
iterator it_    = overlap.first,
last_  = prior(overlap.second);
insert_main(inter_val, co_val, it_, last_);
return it_;
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline void interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::erase_rest(interval_type& inter_val, const CodomainT& co_val, 
iterator& it_, const iterator& last_)
{
while(it_ != last_)
if((*it_).second == co_val)
this->_map.erase(it_++); 
else it_++;

if((*it_).second == co_val)
{
interval_type right_resid = left_subtract((*it_).first, inter_val);
if(icl::is_empty(right_resid))
this->_map.erase(it_);
else
const_cast<interval_type&>((*it_).first) = right_resid;
}
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline SubType& interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::erase(const segment_type& minuend)
{
interval_type inter_val = minuend.first;
if(icl::is_empty(inter_val)) 
return *that();

const codomain_type& co_val = minuend.second;
if(on_codomain_absorbtion::is_absorbable(co_val))
return *that();

std::pair<iterator,iterator> exterior = equal_range(inter_val);
if(exterior.first == exterior.second)
return *that();

iterator first_ = exterior.first, end_ = exterior.second, 
last_  = cyclic_prior(*this, end_);
iterator second_= first_; ++second_;

if(first_ == last_) 
{   
interval_type right_resid = left_subtract((*first_).first, inter_val);

if((*first_).second == co_val)
{   
interval_type left_resid = right_subtract((*first_).first, inter_val);
if(!icl::is_empty(left_resid)) 
{                              
const_cast<interval_type&>((*first_).first) = left_resid;
if(!icl::is_empty(right_resid))
this->_map.insert(first_, value_type(right_resid, co_val));
}
else if(!icl::is_empty(right_resid))
const_cast<interval_type&>((*first_).first) = right_resid;
else
this->_map.erase(first_);
}
}
else
{
if((*first_).second == co_val)
{
interval_type left_resid = right_subtract((*first_).first, inter_val);
if(icl::is_empty(left_resid))
this->_map.erase(first_);
else
const_cast<interval_type&>((*first_).first) = left_resid;
}

erase_rest(inter_val, co_val, second_, last_);
}

return *that();
}

template <class SubType, class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc>
inline SubType& interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc>
::erase(const interval_type& minuend)
{
if(icl::is_empty(minuend)) 
return *that();

std::pair<iterator, iterator> exterior = equal_range(minuend);
if(exterior.first == exterior.second)
return *that();

iterator first_ = exterior.first,
end_   = exterior.second,
last_  = prior(end_);

interval_type left_resid  = right_subtract((*first_).first, minuend);
interval_type right_resid =  left_subtract(last_ ->first, minuend);

if(first_ == last_ )
if(!icl::is_empty(left_resid))
{
const_cast<interval_type&>((*first_).first) = left_resid;
if(!icl::is_empty(right_resid))
this->_map.insert(first_, value_type(right_resid, (*first_).second));
}
else if(!icl::is_empty(right_resid))
const_cast<interval_type&>((*first_).first) = left_subtract((*first_).first, minuend);
else
this->_map.erase(first_);
else
{   
iterator second_= first_; ++second_;

iterator start_ = icl::is_empty(left_resid)? first_: second_;
iterator stop_  = icl::is_empty(right_resid)? end_  : last_ ;
this->_map.erase(start_, stop_); 

if(!icl::is_empty(left_resid))
const_cast<interval_type&>((*first_).first) = left_resid;

if(!icl::is_empty(right_resid))
const_cast<interval_type&>(last_ ->first) = right_resid;
}

return *that();
}

template 
<
class SubType,
class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE)  Interval, ICL_ALLOC Alloc
>
struct is_map<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> >
{ 
typedef is_map<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> > type;
BOOST_STATIC_CONSTANT(bool, value = true); 
};

template 
<
class SubType,
class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE)  Interval, ICL_ALLOC Alloc
>
struct has_inverse<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> >
{ 
typedef has_inverse<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> > type;
BOOST_STATIC_CONSTANT(bool, value = (has_inverse<CodomainT>::value)); 
};

template 
<
class SubType,
class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE)  Interval, ICL_ALLOC Alloc
>
struct is_interval_container<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> >
{ 
typedef is_interval_container<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> > type;
BOOST_STATIC_CONSTANT(bool, value = true); 
};

template 
<
class SubType,
class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE)  Interval, ICL_ALLOC Alloc
>
struct absorbs_identities<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> >
{
typedef absorbs_identities<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> > type;
BOOST_STATIC_CONSTANT(bool, value = (Traits::absorbs_identities)); 
};

template 
<
class SubType,
class DomainT, class CodomainT, class Traits, ICL_COMPARE Compare, ICL_COMBINE Combine, ICL_SECTION Section, ICL_INTERVAL(ICL_COMPARE) Interval, ICL_ALLOC Alloc
>
struct is_total<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> >
{
typedef is_total<icl::interval_base_map<SubType,DomainT,CodomainT,Traits,Compare,Combine,Section,Interval,Alloc> > type;
BOOST_STATIC_CONSTANT(bool, value = (Traits::is_total)); 
};



}} 

#endif


