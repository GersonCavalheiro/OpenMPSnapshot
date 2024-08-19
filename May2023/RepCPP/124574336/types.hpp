#ifndef BOOST_LOCALE_BOUNDARY_TYPES_HPP_INCLUDED
#define BOOST_LOCALE_BOUNDARY_TYPES_HPP_INCLUDED

#include <boost/locale/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif


namespace boost {

namespace locale {

namespace boundary {

enum boundary_type {
character,  
word,       
sentence,   
line        
};

typedef uint32_t rule_type;

static const rule_type
word_none       =  0x0000F,   
word_number     =  0x000F0,   
word_letter     =  0x00F00,   
word_kana       =  0x0F000,   
word_ideo       =  0xF0000,   
word_any        =  0xFFFF0,   
word_letters    =  0xFFF00,   
word_kana_ideo  =  0xFF000,   
word_mask       =  0xFFFFF;   

static const rule_type 
line_soft       =  0x0F,   
line_hard       =  0xF0,   
line_any        =  0xFF,   
line_mask       =  0xFF;   


static const rule_type
sentence_term   =  0x0F,    
sentence_sep    =  0xF0,    
sentence_any    =  0xFF,    
sentence_mask   =  0xFF;    


static const rule_type
character_any   =  0xF,     
character_mask  =  0xF;     


inline rule_type boundary_rule(boundary_type t)
{
switch(t) {
case character: return character_mask;
case word:      return word_mask;
case sentence:  return sentence_mask;
case line:      return line_mask;
default:        return 0;
}
}


} 
} 
} 


#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
