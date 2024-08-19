

#ifndef MU_PARSER_TOKEN_H
#define MU_PARSER_TOKEN_H

#include <string>
#include <stack>
#include <vector>
#include <memory>
#include <utility>
#include <type_traits>
#include <cstddef>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif

#include "muParserError.h"
#include "muParserCallback.h"



namespace mu
{
template <std::size_t NbParams> struct TplCallType;
template <> struct TplCallType<0> { using fun_type = fun_type0; using fun_userdata_type = fun_userdata_type0; using bulkfun_type = bulkfun_type0; using bulkfun_userdata_type = bulkfun_userdata_type0; };
template <> struct TplCallType<1> { using fun_type = fun_type1; using fun_userdata_type = fun_userdata_type1; using bulkfun_type = bulkfun_type1; using bulkfun_userdata_type = bulkfun_userdata_type1; using strfun_type = strfun_type1; using strfun_userdata_type = strfun_userdata_type1; };
template <> struct TplCallType<2> { using fun_type = fun_type2; using fun_userdata_type = fun_userdata_type2; using bulkfun_type = bulkfun_type2; using bulkfun_userdata_type = bulkfun_userdata_type2; using strfun_type = strfun_type2; using strfun_userdata_type = strfun_userdata_type2; };
template <> struct TplCallType<3> { using fun_type = fun_type3; using fun_userdata_type = fun_userdata_type3; using bulkfun_type = bulkfun_type3; using bulkfun_userdata_type = bulkfun_userdata_type3; using strfun_type = strfun_type3; using strfun_userdata_type = strfun_userdata_type3; };
template <> struct TplCallType<4> { using fun_type = fun_type4; using fun_userdata_type = fun_userdata_type4; using bulkfun_type = bulkfun_type4; using bulkfun_userdata_type = bulkfun_userdata_type4; using strfun_type = strfun_type4; using strfun_userdata_type = strfun_userdata_type4; };
template <> struct TplCallType<5> { using fun_type = fun_type5; using fun_userdata_type = fun_userdata_type5; using bulkfun_type = bulkfun_type5; using bulkfun_userdata_type = bulkfun_userdata_type5; using strfun_type = strfun_type5; using strfun_userdata_type = strfun_userdata_type5; };
template <> struct TplCallType<6> { using fun_type = fun_type6; using fun_userdata_type = fun_userdata_type6; using bulkfun_type = bulkfun_type6; using bulkfun_userdata_type = bulkfun_userdata_type6; using strfun_type = strfun_type6; using strfun_userdata_type = strfun_userdata_type6; };
template <> struct TplCallType<7> { using fun_type = fun_type7; using fun_userdata_type = fun_userdata_type7; using bulkfun_type = bulkfun_type7; using bulkfun_userdata_type = bulkfun_userdata_type7; };
template <> struct TplCallType<8> { using fun_type = fun_type8; using fun_userdata_type = fun_userdata_type8; using bulkfun_type = bulkfun_type8; using bulkfun_userdata_type = bulkfun_userdata_type8; };
template <> struct TplCallType<9> { using fun_type = fun_type9; using fun_userdata_type = fun_userdata_type9; using bulkfun_type = bulkfun_type9; using bulkfun_userdata_type = bulkfun_userdata_type9; };
template <> struct TplCallType<10> { using fun_type = fun_type10; using fun_userdata_type = fun_userdata_type10; using bulkfun_type = bulkfun_type10; using bulkfun_userdata_type = bulkfun_userdata_type10; };

struct generic_callable_type
{

erased_fun_type _pRawFun;
void*           _pUserData;

template <std::size_t NbParams, typename... Args>
value_type call_fun(Args&&... args) const
{
static_assert(NbParams == sizeof...(Args), "mismatch between NbParams and Args");
if (_pUserData == nullptr) 
{
auto fun_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::fun_type>(_pRawFun);
return (*fun_typed_ptr)(std::forward<Args>(args)...);
} 
else 
{
auto fun_userdata_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::fun_userdata_type>(_pRawFun);
return (*fun_userdata_typed_ptr)(_pUserData, std::forward<Args>(args)...);
}
}

template <std::size_t NbParams, typename... Args>
value_type call_bulkfun(Args&&... args) const
{
static_assert(NbParams == sizeof...(Args) - 2, "mismatch between NbParams and Args");
if (_pUserData == nullptr) {
auto bulkfun_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::bulkfun_type>(_pRawFun);
return (*bulkfun_typed_ptr)(std::forward<Args>(args)...);
} else {
auto bulkfun_userdata_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::bulkfun_userdata_type>(_pRawFun);
return (*bulkfun_userdata_typed_ptr)(_pUserData, std::forward<Args>(args)...);
}
}

value_type call_multfun(const value_type* a_afArg, int a_iArgc) const
{
if (_pUserData == nullptr) {
auto multfun_typed_ptr = reinterpret_cast<multfun_type>(_pRawFun);
return (*multfun_typed_ptr)(a_afArg, a_iArgc);
} else {
auto multfun_userdata_typed_ptr = reinterpret_cast<multfun_userdata_type>(_pRawFun);
return (*multfun_userdata_typed_ptr)(_pUserData, a_afArg, a_iArgc);
}
}

template <std::size_t NbParams, typename... Args>
value_type call_strfun(Args&&... args) const
{
static_assert(NbParams == sizeof...(Args), "mismatch between NbParams and Args");
if (_pUserData == nullptr) 
{
auto strfun_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::strfun_type>(_pRawFun);
return (*strfun_typed_ptr)(std::forward<Args>(args)...);
} 
else 
{
auto strfun_userdata_typed_ptr = reinterpret_cast<typename TplCallType<NbParams>::strfun_userdata_type>(_pRawFun);
return (*strfun_userdata_typed_ptr)(_pUserData, std::forward<Args>(args)...);
}
}

bool operator==(generic_callable_type other) const 
{
return _pRawFun == other._pRawFun && _pUserData == other._pUserData; 
}

explicit operator bool() const 
{
return _pRawFun != nullptr; 
}

bool operator==(std::nullptr_t) const 
{
return _pRawFun == nullptr; 
}

bool operator!=(std::nullptr_t) const 
{
return _pRawFun != nullptr; 
}
};

static_assert(std::is_trivial<generic_callable_type>::value, "generic_callable_type shall be trivial");
static_assert(std::is_standard_layout<generic_callable_type>::value, "generic_callable_type shall have standard layout");


template<typename TBase, typename TString>
class ParserToken final
{
private:

ECmdCode  m_iCode;  
ETypeCode m_iType;
void* m_pTok;		
int  m_iIdx;		
TString m_strTok;   
TString m_strVal;   
value_type m_fVal;  
std::unique_ptr<ParserCallback> m_pCallback;

public:


ParserToken()
:m_iCode(cmUNKNOWN)
, m_iType(tpVOID)
, m_pTok(0)
, m_iIdx(-1)
, m_strTok()
, m_strVal()
, m_fVal(0)
, m_pCallback()
{}


ParserToken(const ParserToken& a_Tok)
{
Assign(a_Tok);
}



ParserToken& operator=(const ParserToken& a_Tok)
{
Assign(a_Tok);
return *this;
}



void Assign(const ParserToken& a_Tok)
{
m_iCode = a_Tok.m_iCode;
m_pTok = a_Tok.m_pTok;
m_strTok = a_Tok.m_strTok;
m_iIdx = a_Tok.m_iIdx;
m_strVal = a_Tok.m_strVal;
m_iType = a_Tok.m_iType;
m_fVal = a_Tok.m_fVal;
m_pCallback.reset(a_Tok.m_pCallback.get() ? a_Tok.m_pCallback->Clone() : 0);
}


ParserToken& Set(ECmdCode a_iType, const TString& a_strTok = TString())
{
MUP_ASSERT(a_iType != cmVAR);
MUP_ASSERT(a_iType != cmVAL);
MUP_ASSERT(a_iType != cmFUNC);

m_iCode = a_iType;
m_iType = tpVOID;
m_pTok = 0;
m_strTok = a_strTok;
m_iIdx = -1;

return *this;
}


ParserToken& Set(const ParserCallback& a_pCallback, const TString& a_sTok)
{
MUP_ASSERT(a_pCallback.IsValid());

m_iCode = a_pCallback.GetCode();
m_iType = tpVOID;
m_strTok = a_sTok;
m_pCallback.reset(new ParserCallback(a_pCallback));

m_pTok = 0;
m_iIdx = -1;

return *this;
}


ParserToken& SetVal(TBase a_fVal, const TString& a_strTok = TString())
{
m_iCode = cmVAL;
m_iType = tpDBL;
m_fVal = a_fVal;
m_strTok = a_strTok;
m_iIdx = -1;

m_pTok = 0;
m_pCallback.reset(0);

return *this;
}


ParserToken& SetVar(TBase* a_pVar, const TString& a_strTok)
{
m_iCode = cmVAR;
m_iType = tpDBL;
m_strTok = a_strTok;
m_iIdx = -1;
m_pTok = (void*)a_pVar;
m_pCallback.reset(0);
return *this;
}


ParserToken& SetString(const TString& a_strTok, std::size_t a_iSize)
{
m_iCode = cmSTRING;
m_iType = tpSTR;
m_strTok = a_strTok;
m_iIdx = static_cast<int>(a_iSize);

m_pTok = 0;
m_pCallback.reset(0);
return *this;
}


void SetIdx(int a_iIdx)
{
if (m_iCode != cmSTRING || a_iIdx < 0)
throw ParserError(ecINTERNAL_ERROR);

m_iIdx = a_iIdx;
}


int GetIdx() const
{
if (m_iIdx < 0 || m_iCode != cmSTRING)
throw ParserError(ecINTERNAL_ERROR);

return m_iIdx;
}


ECmdCode GetCode() const
{
if (m_pCallback.get())
{
return m_pCallback->GetCode();
}
else
{
return m_iCode;
}
}

ETypeCode GetType() const
{
if (m_pCallback.get())
{
return m_pCallback->GetType();
}
else
{
return m_iType;
}
}

int GetPri() const
{
if (!m_pCallback.get())
throw ParserError(ecINTERNAL_ERROR);

if (m_pCallback->GetCode() != cmOPRT_BIN && m_pCallback->GetCode() != cmOPRT_INFIX)
throw ParserError(ecINTERNAL_ERROR);

return m_pCallback->GetPri();
}

EOprtAssociativity GetAssociativity() const
{
if (m_pCallback.get() == nullptr || m_pCallback->GetCode() != cmOPRT_BIN)
throw ParserError(ecINTERNAL_ERROR);

return m_pCallback->GetAssociativity();
}


generic_callable_type GetFuncAddr() const
{
return (m_pCallback.get())
? generic_callable_type{(erased_fun_type)m_pCallback->GetAddr(),
m_pCallback->GetUserData()}
: generic_callable_type{};
}


TBase GetVal() const
{
switch (m_iCode)
{
case cmVAL:  return m_fVal;
case cmVAR:  return *((TBase*)m_pTok);
default:     throw ParserError(ecVAL_EXPECTED);
}
}


TBase* GetVar() const
{
if (m_iCode != cmVAR)
throw ParserError(ecINTERNAL_ERROR);

return (TBase*)m_pTok;
}


int GetArgCount() const
{
MUP_ASSERT(m_pCallback.get());

if (!m_pCallback->IsValid())
throw ParserError(ecINTERNAL_ERROR);

return m_pCallback->GetArgc();
}


bool IsOptimizable() const
{
return m_pCallback->IsValid() && m_pCallback->IsOptimizable();
}


const TString& GetAsString() const
{
return m_strTok;
}
};
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif
