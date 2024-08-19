

#ifndef MUP_DEF_H
#define MUP_DEF_H

#include <iostream>
#include <string>
#include <sstream>
#include <map>

#include "muParserFixes.h"




#define MUP_BASETYPE double



#if defined(_UNICODE)

#define MUP_STRING_TYPE std::wstring

#if !defined(_T)
#define _T(x) L##x
#endif 
#else
#ifndef _T
#define _T(x) x
#endif


#define MUP_STRING_TYPE std::string
#endif


#define MUP_ASSERT(COND)											\
if (!(COND))											\
{														\
stringstream_type ss;									\
ss << _T("Assertion \"") _T(#COND) _T("\" failed: ")	\
<< __FILE__ << _T(" line ")						\
<< __LINE__ << _T(".");							\
throw ParserError( ecINTERNAL_ERROR, -1, ss.str());   \
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif


namespace mu
{
#if defined(_UNICODE)


inline std::wostream& console()
{
return std::wcout;
}


inline std::wistream& console_in()
{
return std::wcin;
}

#else


inline std::ostream& console()
{
return std::cout;
}


inline std::istream& console_in()
{
return std::cin;
}

#endif


enum ECmdCode
{
cmLE = 0,			
cmGE = 1,			
cmNEQ = 2,			
cmEQ = 3,			
cmLT = 4,			
cmGT = 5,			
cmADD = 6,			
cmSUB = 7,			
cmMUL = 8,			
cmDIV = 9,			
cmPOW = 10,			
cmLAND = 11,
cmLOR = 12,
cmASSIGN = 13,		
cmBO = 14,			
cmBC = 15,			
cmIF = 16,			
cmELSE = 17,		
cmENDIF = 18,		
cmARG_SEP = 19,		
cmVAR = 20,			
cmVAL = 21,			

cmVARPOW2 = 22,
cmVARPOW3 = 23,
cmVARPOW4 = 24,
cmVARMUL = 25,

cmFUNC = 26,		
cmFUNC_STR,			
cmFUNC_BULK,		
cmSTRING,			
cmOPRT_BIN,			
cmOPRT_POSTFIX,		
cmOPRT_INFIX,		
cmEND,				
cmUNKNOWN			
};


enum ETypeCode
{
tpSTR = 0,     
tpDBL = 1,     
tpVOID = 2      
};


enum EParserVersionInfo
{
pviBRIEF,
pviFULL
};



enum EOprtAssociativity
{
oaLEFT = 0,
oaRIGHT = 1,
oaNONE = 2
};



enum EOprtPrecedence
{
prLOR = 1,		
prLAND = 2,		
prBOR = 3,      
prBAND = 4,     
prCMP = 5,		
prADD_SUB = 6,	
prMUL_DIV = 7,	
prPOW = 8,		

prINFIX = 7,	
prPOSTFIX = 7	
};



enum EErrorCodes
{
ecUNEXPECTED_OPERATOR = 0,	
ecUNASSIGNABLE_TOKEN = 1,	
ecUNEXPECTED_EOF = 2,		
ecUNEXPECTED_ARG_SEP = 3,	
ecUNEXPECTED_ARG = 4,		
ecUNEXPECTED_VAL = 5,		
ecUNEXPECTED_VAR = 6,		
ecUNEXPECTED_PARENS = 7,	
ecUNEXPECTED_STR = 8,		
ecSTRING_EXPECTED = 9,		
ecVAL_EXPECTED = 10,		
ecMISSING_PARENS = 11,		
ecUNEXPECTED_FUN = 12,		
ecUNTERMINATED_STRING = 13,	
ecTOO_MANY_PARAMS = 14,		
ecTOO_FEW_PARAMS = 15,		
ecOPRT_TYPE_CONFLICT = 16,	
ecSTR_RESULT = 17,			

ecINVALID_NAME = 18,			
ecINVALID_BINOP_IDENT = 19,		
ecINVALID_INFIX_IDENT = 20,		
ecINVALID_POSTFIX_IDENT = 21,	

ecBUILTIN_OVERLOAD = 22, 
ecINVALID_FUN_PTR = 23, 
ecINVALID_VAR_PTR = 24, 
ecEMPTY_EXPRESSION = 25, 
ecNAME_CONFLICT = 26, 
ecOPT_PRI = 27, 
ecDOMAIN_ERROR = 28, 
ecDIV_BY_ZERO = 29, 
ecGENERIC = 30, 
ecLOCALE = 31, 

ecUNEXPECTED_CONDITIONAL = 32,
ecMISSING_ELSE_CLAUSE = 33,
ecMISPLACED_COLON = 34,

ecUNREASONABLE_NUMBER_OF_COMPUTATIONS = 35,

ecIDENTIFIER_TOO_LONG = 36, 

ecEXPRESSION_TOO_LONG = 37, 

ecINVALID_CHARACTERS_FOUND = 38,

ecINTERNAL_ERROR = 39,    

ecCOUNT,                      
ecUNDEFINED = -1  
};



typedef MUP_BASETYPE value_type;


typedef MUP_STRING_TYPE string_type;


typedef string_type::value_type char_type;


typedef std::basic_stringstream<char_type, std::char_traits<char_type>, std::allocator<char_type> > stringstream_type;



typedef std::map<string_type, value_type*> varmap_type;


typedef std::map<string_type, value_type> valmap_type;


typedef std::map<string_type, std::size_t> strmap_type;



typedef void(*erased_fun_type)();


typedef value_type(*fun_type0)();


typedef value_type(*fun_type1)(value_type);


typedef value_type(*fun_type2)(value_type, value_type);


typedef value_type(*fun_type3)(value_type, value_type, value_type);


typedef value_type(*fun_type4)(value_type, value_type, value_type, value_type);


typedef value_type(*fun_type5)(value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_type6)(value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_type7)(value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_type8)(value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_type9)(value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_type10)(value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type0)(void*);


typedef value_type(*fun_userdata_type1)(void*, value_type);


typedef value_type(*fun_userdata_type2)(void*, value_type, value_type);


typedef value_type(*fun_userdata_type3)(void*, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type4)(void*, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type5)(void*, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type6)(void*, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type7)(void*, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type8)(void*, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type9)(void*, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*fun_userdata_type10)(void*, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type0)(int, int);


typedef value_type(*bulkfun_type1)(int, int, value_type);


typedef value_type(*bulkfun_type2)(int, int, value_type, value_type);


typedef value_type(*bulkfun_type3)(int, int, value_type, value_type, value_type);


typedef value_type(*bulkfun_type4)(int, int, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type5)(int, int, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type6)(int, int, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type7)(int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type8)(int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type9)(int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_type10)(int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type0)(void*, int, int);


typedef value_type(*bulkfun_userdata_type1)(void*, int, int, value_type);


typedef value_type(*bulkfun_userdata_type2)(void*, int, int, value_type, value_type);


typedef value_type(*bulkfun_userdata_type3)(void*, int, int, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type4)(void*, int, int, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type5)(void*, int, int, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type6)(void*, int, int, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type7)(void*, int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type8)(void*, int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type9)(void*, int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*bulkfun_userdata_type10)(void*, int, int, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*multfun_type)(const value_type*, int);


typedef value_type(*multfun_userdata_type)(void*, const value_type*, int);


typedef value_type(*strfun_type1)(const char_type*);


typedef value_type(*strfun_type2)(const char_type*, value_type);


typedef value_type(*strfun_type3)(const char_type*, value_type, value_type);


typedef value_type(*strfun_type4)(const char_type*, value_type, value_type, value_type);


typedef value_type(*strfun_type5)(const char_type*, value_type, value_type, value_type, value_type);


typedef value_type(*strfun_type6)(const char_type*, value_type, value_type, value_type, value_type, value_type);


typedef value_type(*strfun_userdata_type1)(void*, const char_type*);


typedef value_type(*strfun_userdata_type2)(void*, const char_type*, value_type);


typedef value_type(*strfun_userdata_type3)(void*, const char_type*, value_type, value_type);


typedef value_type(*strfun_userdata_type4)(void*, const char_type*, value_type, value_type, value_type);


typedef value_type(*strfun_userdata_type5)(void*, const char_type*, value_type, value_type, value_type, value_type);


typedef value_type(*strfun_userdata_type6)(void*, const char_type*, value_type, value_type, value_type, value_type, value_type);


typedef int (*identfun_type)(const char_type* sExpr, int* nPos, value_type* fVal);


typedef value_type* (*facfun_type)(const char_type*, void*);

static const int MaxLenExpression = 20000;
static const int MaxLenIdentifier = 100;
static const string_type ParserVersion = string_type(_T("2.3.4 (Develop)"));
static const string_type ParserVersionDate = string_type(_T("20230307"));
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

