
#ifndef BOOST_SPIRIT_REPOSITORY_QI_OPERATOR_DETAIL_KEYWORDS_HPP
#define BOOST_SPIRIT_REPOSITORY_QI_OPERATOR_DETAIL_KEYWORDS_HPP

#if defined(_MSC_VER)
#pragma once
#endif
#include <boost/fusion/include/nview.hpp>
#include <boost/spirit/home/qi/string/lit.hpp>
#include <boost/fusion/include/at.hpp>
namespace boost { namespace spirit { namespace repository { namespace qi { namespace detail {
template<typename T>
struct is_distinct : T::distinct { };

template<typename T, typename Action>
struct is_distinct< spirit::qi::action<T,Action> > : T::distinct { };

template<typename T>
struct is_distinct< spirit::qi::hold_directive<T> > : T::distinct { };



template < typename Elements, typename Iterator ,typename Context ,typename Skipper
,typename Flags ,typename Counters ,typename Attribute, typename NoCasePass>
struct parse_dispatcher
: public boost::static_visitor<bool>
{

typedef Iterator iterator_type;
typedef Context context_type;
typedef Skipper skipper_type;
typedef Elements elements_type;

typedef typename add_reference<Attribute>::type attr_reference;
public:
parse_dispatcher(const Elements &elements,Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Flags &flags, Counters &counters, attr_reference attr) :
elements(elements), first(first), last(last)
, context(context), skipper(skipper)
, flags(flags),counters(counters), attr(attr)
{}

template<typename T> bool operator()(T& idx) const
{
return call(idx,typename traits::not_is_unused<Attribute>::type());
}

template <typename Subject,typename Index>
bool call_subject_unused(
Subject const &subject, Iterator &first, Iterator const &last
, Context& context, Skipper const& skipper
, Index&  ) const
{
Iterator save = first;
skipper_keyword_marker<Skipper,NoCasePass>
marked_skipper(skipper,flags[Index::value],counters[Index::value]);

if(subject.parse(first,last,context,marked_skipper,unused))
{
return true;
}
first = save;
return false;
}


template <typename Subject,typename Index>
bool call_subject(
Subject const &subject, Iterator &first, Iterator const &last
, Context& context, Skipper const& skipper
, Index&  ) const
{

Iterator save = first;
skipper_keyword_marker<Skipper,NoCasePass> 
marked_skipper(skipper,flags[Index::value],counters[Index::value]);
if(subject.parse(first,last,context,marked_skipper,fusion::at_c<Index::value>(attr)))
{
return true;
}
first = save;
return false;
}

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif
template <typename T> bool call(T &idx, mpl::false_) const{

typedef typename mpl::at<Elements,T>::type ElementType;
if(
(!is_distinct<ElementType>::value)
|| skipper.parse(first,last,unused,unused,unused)
){
spirit::qi::skip_over(first, last, skipper);
return call_subject_unused(fusion::at_c<T::value>(elements), first, last, context, skipper, idx );
}
return false;
}
template <typename T> bool call(T &idx, mpl::true_) const{
typedef typename mpl::at<Elements,T>::type ElementType;
if(
(!is_distinct<ElementType>::value)
|| skipper.parse(first,last,unused,unused,unused)
){
return call_subject(fusion::at_c<T::value>(elements), first, last, context, skipper, idx);
}
return false;
}
#if defined(_MSC_VER)
# pragma warning(pop)
#endif

const Elements &elements;
Iterator &first;
const Iterator &last;
Context & context;
const Skipper &skipper;
Flags &flags;
Counters &counters;
attr_reference attr;
};
template <typename Elements, typename StringKeywords, typename IndexList, typename FlagsType, typename Modifiers>
struct string_keywords
{
typedef typename
spirit::detail::as_variant<
IndexList >::type        parser_index_type;

template <typename Sequence >
struct build_char_type_sequence
{
struct element_char_type
{
template <typename T>
struct result;

template <typename F, typename Element>
struct result<F(Element)>
{
typedef typename Element::char_type type;

};
template <typename F, typename Element,typename Action>
struct result<F(spirit::qi::action<Element,Action>) >
{
typedef typename Element::char_type type;
};
template <typename F, typename Element>
struct result<F(spirit::qi::hold_directive<Element>)>
{
typedef typename Element::char_type type;
};

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename Element>
typename result<element_char_type(Element)>::type
operator()(Element&&) const;
#endif
};

typedef typename
fusion::result_of::transform<Sequence, element_char_type>::type
type;
};


template <typename Sequence>
struct get_keyword_char_type
{
typedef typename
mpl::fold<
Sequence, mpl::vector<>,
mpl::if_<
mpl::contains<mpl::_1, mpl::_2>,
mpl::_1, mpl::push_back<mpl::_1, mpl::_2>
>
>::type
no_duplicate_char_types;

BOOST_MPL_ASSERT_RELATION( mpl::size<no_duplicate_char_types>::value, ==, 1 );

typedef typename mpl::front<no_duplicate_char_types>::type type;

};

typedef typename build_char_type_sequence< StringKeywords >::type char_types;
typedef typename get_keyword_char_type<
typename mpl::if_<
mpl::equal_to<
typename mpl::size < char_types >::type
, mpl::int_<0>
>
, mpl::vector< boost::spirit::standard::char_type >
, char_types >::type
>::type  char_type;

typedef spirit::qi::tst< char_type, parser_index_type> keywords_type;

template <typename CharEncoding>
struct no_case_filter
{
char_type operator()(char_type ch) const
{
return static_cast<char_type>(CharEncoding::tolower(ch));
}
};

template <typename Sequence >
struct build_case_type_sequence
{
struct element_case_type
{
template <typename T>
struct result;

template <typename F, typename Element>
struct result<F(Element)>
{
typedef typename Element::no_case_keyword type;

};
template <typename F, typename Element,typename Action>
struct result<F(spirit::qi::action<Element,Action>) >
{
typedef typename Element::no_case_keyword type;
};
template <typename F, typename Element>
struct result<F(spirit::qi::hold_directive<Element>)>
{
typedef typename Element::no_case_keyword type;
};

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename Element>
typename result<element_case_type(Element)>::type
operator()(Element&&) const;
#endif
};

typedef typename
fusion::result_of::transform<Sequence, element_case_type>::type
type;
};

template <typename Sequence,typename CaseType>
struct get_nb_case_types
{
typedef typename
mpl::count_if<
Sequence, mpl::equal_to<mpl::_,CaseType>
>::type type;


};
typedef typename build_case_type_sequence< StringKeywords >::type case_type_sequence;
typedef typename get_nb_case_types<case_type_sequence,mpl::true_>::type ikwd_count;
typedef typename get_nb_case_types<case_type_sequence,mpl::false_>::type kwd_count;
typedef typename mpl::size<IndexList>::type nb_elements;
typedef typename mpl::and_<
typename mpl::greater< nb_elements, mpl::int_<0> >::type
, typename mpl::equal_to< ikwd_count, nb_elements>::type
>::type all_ikwd;

typedef typename mpl::and_<
typename mpl::greater< nb_elements, mpl::int_<0> >::type
, typename mpl::equal_to< kwd_count, nb_elements>::type
>::type all_kwd;

typedef typename mpl::or_< all_kwd, all_ikwd >::type all_directives_of_same_type;

typedef has_modifier<Modifiers, spirit::tag::char_code_base<spirit::tag::no_case> > no_case_modifier;

typedef typename mpl::or_<
no_case_modifier,
mpl::and_<
all_directives_of_same_type
,all_ikwd
>
>::type
no_case;

typedef no_case_filter<
typename spirit::detail::get_encoding_with_case<
Modifiers
, char_encoding::standard
, no_case::value>::type>
nc_filter;
typedef typename mpl::if_<
no_case
, nc_filter
, spirit::qi::tst_pass_through >::type
first_pass_filter_type;

typedef typename mpl::or_<
all_directives_of_same_type
, no_case_modifier
>::type requires_one_pass;


struct keyword_entry_adder
{
typedef int result_type;

keyword_entry_adder(shared_ptr<keywords_type> lookup,FlagsType &flags, Elements &elements) :
lookup(lookup)
,flags(flags)
,elements(elements)
{}

template <typename T>
int operator()(const T &index) const
{
return call(fusion::at_c<T::value>(elements),index);
}

template <typename T, typename Position, typename Action>
int call(const spirit::qi::action<T,Action> &parser, const Position position ) const
{

lookup->add(
traits::get_begin<char_type>(get_string(parser.subject.keyword)),
traits::get_end<char_type>(get_string(parser.subject.keyword)),
position
);
flags[Position::value]=parser.subject.iter.flag_init();
return 0;
}

template <typename T, typename Position>
int call( const T & parser, const Position position) const
{
lookup->add(
traits::get_begin<char_type>(get_string(parser.keyword)),
traits::get_end<char_type>(get_string(parser.keyword)),
position
);
flags[Position::value]=parser.iter.flag_init();
return 0;
}

template <typename T, typename Position>
int call( const spirit::qi::hold_directive<T> & parser, const Position position) const
{
lookup->add(
traits::get_begin<char_type>(get_string(parser.subject.keyword)),
traits::get_end<char_type>(get_string(parser.subject.keyword)),
position
);
flags[Position::value]=parser.subject.iter.flag_init();
return 0;
}


template <typename String, bool no_attribute>
const String get_string(const boost::spirit::qi::literal_string<String,no_attribute> &parser) const
{
return parser.str;
}

template <typename String, bool no_attribute>
const typename boost::spirit::qi::no_case_literal_string<String,no_attribute>::string_type &
get_string(const boost::spirit::qi::no_case_literal_string<String,no_attribute> &parser) const
{
return parser.str_lo;
}



shared_ptr<keywords_type> lookup;
FlagsType & flags;
Elements &elements;
};

string_keywords(Elements &elements,FlagsType &flags_init) : lookup(new keywords_type())
{
IndexList indexes;
keyword_entry_adder f1(lookup,flags_init,elements);
fusion::for_each(indexes,f1);

}
template <typename Iterator,typename ParseVisitor, typename Skipper>
bool parse(
Iterator &first,
const Iterator &last,
const ParseVisitor &parse_visitor,
const Skipper &) const
{
if(parser_index_type* val_ptr =
lookup->find(first,last,first_pass_filter_type()))
{                        
if(!apply_visitor(parse_visitor,*val_ptr)){
return false;
}
return true;
}
return false;
}

template <typename Iterator,typename ParseVisitor, typename NoCaseParseVisitor,typename Skipper>
bool parse(
Iterator &first,
const Iterator &last,
const ParseVisitor &parse_visitor,
const NoCaseParseVisitor &no_case_parse_visitor,
const Skipper &) const
{
Iterator saved_first = first;
if(parser_index_type* val_ptr =
lookup->find(first,last,first_pass_filter_type()))
{
if(!apply_visitor(parse_visitor,*val_ptr)){
return false;
}
return true;
}
else if(parser_index_type* val_ptr
= lookup->find(saved_first,last,nc_filter()))
{
first = saved_first;
if(!apply_visitor(no_case_parse_visitor,*val_ptr)){
return false;
}
return true;
}
return false;
}
shared_ptr<keywords_type> lookup;


};

struct empty_keywords_list
{
typedef mpl::true_ requires_one_pass;

empty_keywords_list()
{}
template<typename Elements>
empty_keywords_list(const Elements &)
{}

template<typename Elements, typename FlagsInit>
empty_keywords_list(const Elements &, const FlagsInit &)
{}

template <typename Iterator,typename ParseVisitor, typename NoCaseParseVisitor,typename Skipper>
bool parse(
Iterator &,
const Iterator &,
const ParseVisitor &,
const NoCaseParseVisitor &,
const Skipper &) const
{
return false;
}

template <typename Iterator,typename ParseVisitor, typename Skipper>
bool parse(
Iterator &,
const Iterator &,
const ParseVisitor &,
const Skipper &) const
{
return false;
}

template <typename ParseFunction>
bool parse( ParseFunction & ) const
{
return false;
}
};

template<typename ComplexKeywords>
struct complex_keywords
{
template <typename FlagsType, typename Elements>
struct flag_init_value_setter
{
typedef int result_type;

flag_init_value_setter(Elements &elements,FlagsType &flags)
:flags(flags)
,elements(elements)
{}

template <typename T>
int operator()(const T &index) const
{
return call(fusion::at_c<T::value>(elements),index);
}

template <typename T, typename Position, typename Action>
int call(const spirit::qi::action<T,Action> &parser, const Position  ) const
{
flags[Position::value]=parser.subject.iter.flag_init();
return 0;
}

template <typename T, typename Position>
int call( const T & parser, const Position ) const
{
flags[Position::value]=parser.iter.flag_init();
return 0;
}

template <typename T, typename Position>
int call( const spirit::qi::hold_directive<T> & parser, const Position ) const
{
flags[Position::value]=parser.subject.iter.flag_init();
return 0;
}

FlagsType & flags;
Elements &elements;
};

template <typename Elements, typename Flags>
complex_keywords(Elements &elements, Flags &flags)
{
flag_init_value_setter<Flags,Elements> flag_initializer(elements,flags);
fusion::for_each(complex_keywords_inst,flag_initializer);
}

template <typename ParseFunction>
bool parse( ParseFunction &function ) const
{
return fusion::any(complex_keywords_inst,function);
}

ComplexKeywords complex_keywords_inst;
};
struct register_successful_parse
{
template <typename Subject>
static bool call(Subject const &subject,bool &flag, int &counter)
{
return subject.iter.register_successful_parse(flag,counter);
}
template <typename Subject, typename Action>
static bool call(spirit::qi::action<Subject, Action> const &subject,bool &flag, int &counter)
{
return subject.subject.iter.register_successful_parse(flag,counter);
}
template <typename Subject>
static bool call(spirit::qi::hold_directive<Subject> const &subject,bool &flag, int &counter)
{
return subject.subject.iter.register_successful_parse(flag,counter);
}
};

struct extract_keyword
{
template <typename Subject>
static Subject const& call(Subject const &subject)
{
return subject;
}
template <typename Subject, typename Action>
static Subject const& call(spirit::qi::action<Subject, Action> const &subject)
{
return subject.subject;
}
template <typename Subject>
static Subject const& call(spirit::qi::hold_directive<Subject> const &subject)
{
return subject.subject;
}
};

template <typename ParseDispatcher>
struct complex_kwd_function
{
typedef typename ParseDispatcher::iterator_type Iterator;
typedef typename ParseDispatcher::context_type Context;
typedef typename ParseDispatcher::skipper_type Skipper;
complex_kwd_function(
Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper, ParseDispatcher &dispatcher)
: first(first)
, last(last)
, context(context)
, skipper(skipper)
, dispatcher(dispatcher)
{
}

template <typename Component>
bool operator()(Component const& component)
{
Iterator save = first;
if(
extract_keyword::call(
fusion::at_c<
Component::value
,typename ParseDispatcher::elements_type
>(dispatcher.elements)
)
.keyword.parse(
first
,last
,context
,skipper
,unused)
)
{
if(!dispatcher(component)){
first = save;
return false;
}
return true;
}
return false;
}

Iterator& first;
Iterator const& last;
Context& context;
Skipper const& skipper;
ParseDispatcher const& dispatcher;

BOOST_DELETED_FUNCTION(complex_kwd_function& operator= (complex_kwd_function const&))
};


}}}}}

#endif
