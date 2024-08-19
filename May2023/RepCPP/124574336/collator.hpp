#ifndef BOOST_LOCALE_COLLATOR_HPP_INCLUDED
#define BOOST_LOCALE_COLLATOR_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <locale>


namespace boost {
namespace locale {

class info;



class collator_base {
public:
typedef enum {
primary     = 0, 
secondary   = 1, 
tertiary    = 2, 
quaternary  = 3, 
identical   = 4  
} level_type;
};

template<typename CharType>
class collator : 
public std::collate<CharType>,
public collator_base
{
public:
typedef CharType char_type;
typedef std::basic_string<CharType> string_type;


int compare(level_type level,
char_type const *b1,char_type const *e1,
char_type const *b2,char_type const *e2) const
{
return do_compare(level,b1,e1,b2,e2);
}
string_type transform(level_type level,char_type const *b,char_type const *e) const
{
return do_transform(level,b,e);
}

long hash(level_type level,char_type const *b,char_type const *e) const
{
return do_hash(level,b,e);
}

int compare(level_type level,string_type const &l,string_type const &r) const
{
return do_compare(level,l.data(),l.data()+l.size(),r.data(),r.data()+r.size());
}


long hash(level_type level,string_type const &s) const
{
return do_hash(level,s.data(),s.data()+s.size());
}
string_type transform(level_type level,string_type const &s) const
{
return do_transform(level,s.data(),s.data()+s.size());
}

protected:

collator(size_t refs = 0) : std::collate<CharType>(refs) 
{
}

virtual ~collator()
{
}

virtual int do_compare( char_type const *b1,char_type const *e1,
char_type const *b2,char_type const *e2) const
{
return do_compare(identical,b1,e1,b2,e2);
}
virtual string_type do_transform(char_type const *b,char_type const *e) const
{
return do_transform(identical,b,e);
}
virtual long do_hash(char_type const *b,char_type const *e) const
{
return do_hash(identical,b,e);
}

virtual int do_compare( level_type level,
char_type const *b1,char_type const *e1,
char_type const *b2,char_type const *e2) const = 0;
virtual string_type do_transform(level_type level,char_type const *b,char_type const *e) const = 0;
virtual long do_hash(level_type level,char_type const *b,char_type const *e) const = 0;


};

template<typename CharType,collator_base::level_type default_level = collator_base::identical>
struct comparator
{
public:
comparator(std::locale const &l=std::locale(),collator_base::level_type level=default_level) : 
locale_(l),
level_(level)
{
}

bool operator()(std::basic_string<CharType> const &left,std::basic_string<CharType> const &right) const
{
return std::use_facet<collator<CharType> >(locale_).compare(level_,left,right) < 0;
}
private:
std::locale locale_;
collator_base::level_type level_;
};



} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif
