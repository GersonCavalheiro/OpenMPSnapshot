#ifndef BOOST_LOCALE_CONVERTER_HPP_INCLUDED
#define BOOST_LOCALE_CONVERTER_HPP_INCLUDED

#include <boost/locale/config.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <locale>


namespace boost {
namespace locale {



class converter_base {
public:
typedef enum {
normalization,  
upper_case,     
lower_case,     
case_folding,   
title_case      
} conversion_type;
};

template<typename CharType>
class converter;

#ifdef BOOST_LOCALE_DOXYGEN
template<typename Char>
class BOOST_LOCALE_DECL converter: public converter_base, public std::locale::facet {
public:
static std::locale::id id;

converter(size_t refs = 0) : std::locale::facet(refs)
{
}
virtual std::basic_string<Char> convert(conversion_type how,Char const *begin,Char const *end,int flags = 0) const = 0;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};
#else

template<>
class BOOST_LOCALE_DECL converter<char> : public converter_base, public std::locale::facet {
public:
static std::locale::id id;

converter(size_t refs = 0) : std::locale::facet(refs)
{
}
virtual std::string convert(conversion_type how,char const *begin,char const *end,int flags = 0) const = 0;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};

template<>
class BOOST_LOCALE_DECL converter<wchar_t> : public converter_base, public std::locale::facet {
public:
static std::locale::id id;
converter(size_t refs = 0) : std::locale::facet(refs)
{
}
virtual std::wstring convert(conversion_type how,wchar_t const *begin,wchar_t const *end,int flags = 0) const = 0;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};

#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
template<>
class BOOST_LOCALE_DECL converter<char16_t> : public converter_base, public std::locale::facet {
public:
static std::locale::id id;
converter(size_t refs = 0) : std::locale::facet(refs)
{
}
virtual std::u16string convert(conversion_type how,char16_t const *begin,char16_t const *end,int flags = 0) const = 0; 
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};
#endif

#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
template<>
class BOOST_LOCALE_DECL converter<char32_t> : public converter_base, public std::locale::facet {
public:
static std::locale::id id;
converter(size_t refs = 0) : std::locale::facet(refs)
{
}
virtual std::u32string convert(conversion_type how,char32_t const *begin,char32_t const *end,int flags = 0) const = 0;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};
#endif

#endif


typedef enum {
norm_nfd,   
norm_nfc,   
norm_nfkd,  
norm_nfkc,  
norm_default = norm_nfc, 
} norm_type;

template<typename CharType>
std::basic_string<CharType> normalize(std::basic_string<CharType> const &str,norm_type n=norm_default,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::normalization,str.data(),str.data() + str.size(),n);
}

template<typename CharType>
std::basic_string<CharType> normalize(CharType const *str,norm_type n=norm_default,std::locale const &loc=std::locale())
{
CharType const *end=str;
while(*end)
end++;
return std::use_facet<converter<CharType> >(loc).convert(converter_base::normalization,str,end,n);
}

template<typename CharType>
std::basic_string<CharType> normalize(  CharType const *begin,
CharType const *end,
norm_type n=norm_default,
std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::normalization,begin,end,n);
}



template<typename CharType>
std::basic_string<CharType> to_upper(std::basic_string<CharType> const &str,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::upper_case,str.data(),str.data()+str.size());
}

template<typename CharType>
std::basic_string<CharType> to_upper(CharType const *str,std::locale const &loc=std::locale())
{
CharType const *end=str;
while(*end)
end++;
return std::use_facet<converter<CharType> >(loc).convert(converter_base::upper_case,str,end);
}

template<typename CharType>
std::basic_string<CharType> to_upper(CharType const *begin,CharType const *end,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::upper_case,begin,end);
}



template<typename CharType>
std::basic_string<CharType> to_lower(std::basic_string<CharType> const &str,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::lower_case,str.data(),str.data()+str.size());
}

template<typename CharType>
std::basic_string<CharType> to_lower(CharType const *str,std::locale const &loc=std::locale())
{
CharType const *end=str;
while(*end)
end++;
return std::use_facet<converter<CharType> >(loc).convert(converter_base::lower_case,str,end);
}

template<typename CharType>
std::basic_string<CharType> to_lower(CharType const *begin,CharType const *end,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::lower_case,begin,end);
}


template<typename CharType>
std::basic_string<CharType> to_title(std::basic_string<CharType> const &str,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::title_case,str.data(),str.data()+str.size());
}

template<typename CharType>
std::basic_string<CharType> to_title(CharType const *str,std::locale const &loc=std::locale())
{
CharType const *end=str;
while(*end)
end++;
return std::use_facet<converter<CharType> >(loc).convert(converter_base::title_case,str,end);
}

template<typename CharType>
std::basic_string<CharType> to_title(CharType const *begin,CharType const *end,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::title_case,begin,end);
}



template<typename CharType>
std::basic_string<CharType> fold_case(std::basic_string<CharType> const &str,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::case_folding,str.data(),str.data()+str.size());
}

template<typename CharType>
std::basic_string<CharType> fold_case(CharType const *str,std::locale const &loc=std::locale())
{
CharType const *end=str;
while(*end)
end++;
return std::use_facet<converter<CharType> >(loc).convert(converter_base::case_folding,str,end);
}

template<typename CharType>
std::basic_string<CharType> fold_case(CharType const *begin,CharType const *end,std::locale const &loc=std::locale())
{
return std::use_facet<converter<CharType> >(loc).convert(converter_base::case_folding,begin,end);
}

} 

} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif


#endif



