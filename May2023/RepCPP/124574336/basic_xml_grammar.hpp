#ifndef BOOST_ARCHIVE_BASIC_XML_GRAMMAR_HPP
#define BOOST_ARCHIVE_BASIC_XML_GRAMMAR_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <string>

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/spirit/include/classic_rule.hpp>
#include <boost/spirit/include/classic_chset.hpp>

#include <boost/archive/basic_archive.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/version.hpp>

namespace boost {
namespace archive {


template<class CharType>
class BOOST_SYMBOL_VISIBLE basic_xml_grammar {
public:
struct return_values;
friend struct return_values;

private:
typedef typename std::basic_istream<CharType> IStream;
typedef typename std::basic_string<CharType> StringType;
typedef typename boost::spirit::classic::chset<CharType> chset_t;
typedef typename boost::spirit::classic::chlit<CharType> chlit_t;
typedef typename boost::spirit::classic::scanner<
typename  std::basic_string<CharType>::iterator
> scanner_t;
typedef typename boost::spirit::classic::rule<scanner_t> rule_t;
rule_t
Reference,
Eq,
STag,
ETag,
LetterOrUnderscoreOrColon,
AttValue,
CharRef1,
CharRef2,
CharRef,
AmpRef,
LTRef,
GTRef,
AposRef,
QuoteRef,
CharData,
CharDataChars,
content,
AmpName,
LTName,
GTName,
ClassNameChar,
ClassName,
Name,
XMLDecl,
XMLDeclChars,
DocTypeDecl,
DocTypeDeclChars,
ClassIDAttribute,
ObjectIDAttribute,
ClassNameAttribute,
TrackingAttribute,
VersionAttribute,
UnusedAttribute,
Attribute,
SignatureAttribute,
SerializationWrapper,
NameHead,
NameTail,
AttributeList,
S;

chset_t
BaseChar,
Ideographic,
Char,
Letter,
Digit,
CombiningChar,
Extender,
Sch,
NameChar;

void init_chset();

bool my_parse(
IStream & is,
const rule_t &rule_,
const CharType delimiter = L'>'
) const ;
public:
struct return_values {
StringType object_name;
StringType contents;
int_least16_t class_id;
uint_least32_t object_id;
unsigned int version;
tracking_type tracking_level;
StringType class_name;
return_values() :
version(0),
tracking_level(false)
{}
} rv;
bool parse_start_tag(IStream & is) ;
bool parse_end_tag(IStream & is) const;
bool parse_string(IStream & is, StringType & s) ;
void init(IStream & is);
bool windup(IStream & is);
basic_xml_grammar();
};

} 
} 

#endif 
