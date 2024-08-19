#pragma once

#include <exception>

namespace Xbyak_aarch64 {

enum {
ERR_NONE = 0,
ERR_CODE_IS_TOO_BIG,           
ERR_LABEL_IS_REDEFINED,        
ERR_LABEL_IS_TOO_FAR,          
ERR_LABEL_IS_NOT_FOUND,        
ERR_BAD_PARAMETER,             
ERR_CANT_PROTECT,              
ERR_OFFSET_IS_TOO_BIG,         
ERR_CANT_ALLOC,                
ERR_LABEL_ISNOT_SET_BY_L,      
ERR_LABEL_IS_ALREADY_SET_BY_L, 
ERR_INTERNAL,                  
ERR_ILLEGAL_REG_IDX,           
ERR_ILLEGAL_REG_ELEM_IDX,      
ERR_ILLEGAL_PREDICATE_TYPE,    
ERR_ILLEGAL_IMM_RANGE,         
ERR_ILLEGAL_IMM_VALUE,         
ERR_ILLEGAL_IMM_COND,          
ERR_ILLEGAL_SHMOD,             
ERR_ILLEGAL_EXTMOD,            
ERR_ILLEGAL_COND,              
ERR_ILLEGAL_BARRIER_OPT,       
ERR_ILLEGAL_CONST_RANGE,       
ERR_ILLEGAL_CONST_VALUE,       
ERR_ILLEGAL_CONST_COND,        
ERR_ILLEGAL_TYPE,
ERR_BAD_ALIGN,
ERR_BAD_ADDRESSING,
ERR_BAD_SCALE,
ERR_MUNMAP,
};

class Error : public std::exception {
int err_;
const char *msg_;

public:
explicit Error(int err);
operator int() const { return err_; }
const char *what() const throw() { return msg_; }
};

inline const char *ConvertErrorToString(const Error &err) { return err.what(); }

} 
