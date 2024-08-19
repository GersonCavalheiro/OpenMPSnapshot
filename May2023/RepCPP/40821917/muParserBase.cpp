

#include "muParserBase.h"
#include "muParserTemplateMagic.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <deque>
#include <sstream>
#include <locale>
#include <cassert>
#include <cctype>

#ifdef MUP_USE_OPENMP

#include <omp.h>

#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif

using namespace std;



namespace mu
{
std::locale ParserBase::s_locale = std::locale(std::locale::classic(), new change_dec_sep<char_type>('.'));

bool ParserBase::g_DbgDumpCmdCode = false;
bool ParserBase::g_DbgDumpStack = false;


const char_type* ParserBase::c_DefaultOprt[] =
{
_T("<="), _T(">="),  _T("!="),
_T("=="), _T("<"),   _T(">"),
_T("+"),  _T("-"),   _T("*"),
_T("/"),  _T("^"),   _T("&&"),
_T("||"), _T("="),   _T("("),
_T(")"),   _T("?"),  _T(":"), 0
};

const int ParserBase::s_MaxNumOpenMPThreads = 16;


ParserBase::ParserBase()
: m_pParseFormula(&ParserBase::ParseString)
, m_vRPN()
, m_vStringBuf()
, m_pTokenReader()
, m_FunDef()
, m_PostOprtDef()
, m_InfixOprtDef()
, m_OprtDef()
, m_ConstDef()
, m_StrVarDef()
, m_VarDef()
, m_bBuiltInOp(true)
, m_sNameChars()
, m_sOprtChars()
, m_sInfixOprtChars()
, m_vStackBuffer()
, m_nFinalResultIdx(0)
{
InitTokenReader();
}


ParserBase::ParserBase(const ParserBase& a_Parser)
: m_pParseFormula(&ParserBase::ParseString)
, m_vRPN()
, m_vStringBuf()
, m_pTokenReader()
, m_FunDef()
, m_PostOprtDef()
, m_InfixOprtDef()
, m_OprtDef()
, m_ConstDef()
, m_StrVarDef()
, m_VarDef()
, m_bBuiltInOp(true)
, m_sNameChars()
, m_sOprtChars()
, m_sInfixOprtChars()
{
m_pTokenReader.reset(new token_reader_type(this));
Assign(a_Parser);
}

ParserBase::~ParserBase()
{}


ParserBase& ParserBase::operator=(const ParserBase& a_Parser)
{
Assign(a_Parser);
return *this;
}


void ParserBase::Assign(const ParserBase& a_Parser)
{
if (&a_Parser == this)
return;

ReInit();

m_ConstDef = a_Parser.m_ConstDef;         
m_VarDef = a_Parser.m_VarDef;           
m_bBuiltInOp = a_Parser.m_bBuiltInOp;
m_vStringBuf = a_Parser.m_vStringBuf;
m_vStackBuffer = a_Parser.m_vStackBuffer;
m_nFinalResultIdx = a_Parser.m_nFinalResultIdx;
m_StrVarDef = a_Parser.m_StrVarDef;
m_vStringVarBuf = a_Parser.m_vStringVarBuf;
m_pTokenReader.reset(a_Parser.m_pTokenReader->Clone(this));

m_FunDef = a_Parser.m_FunDef;             
m_PostOprtDef = a_Parser.m_PostOprtDef;   
m_InfixOprtDef = a_Parser.m_InfixOprtDef; 
m_OprtDef = a_Parser.m_OprtDef;           

m_sNameChars = a_Parser.m_sNameChars;
m_sOprtChars = a_Parser.m_sOprtChars;
m_sInfixOprtChars = a_Parser.m_sInfixOprtChars;
}


void ParserBase::SetDecSep(char_type cDecSep)
{
char_type cThousandsSep = std::use_facet< change_dec_sep<char_type> >(s_locale).thousands_sep();
s_locale = std::locale(std::locale("C"), new change_dec_sep<char_type>(cDecSep, cThousandsSep));
}


void ParserBase::SetThousandsSep(char_type cThousandsSep)
{
char_type cDecSep = std::use_facet< change_dec_sep<char_type> >(s_locale).decimal_point();
s_locale = std::locale(std::locale("C"), new change_dec_sep<char_type>(cDecSep, cThousandsSep));
}


void ParserBase::ResetLocale()
{
s_locale = std::locale(std::locale("C"), new change_dec_sep<char_type>('.'));
SetArgSep(',');
}


void ParserBase::InitTokenReader()
{
m_pTokenReader.reset(new token_reader_type(this));
}


void ParserBase::ReInit() const
{
m_pParseFormula = &ParserBase::ParseString;
m_vStringBuf.clear();
m_vRPN.clear();
m_pTokenReader->ReInit();
}

void ParserBase::OnDetectVar(string_type* , int& , int& )
{}


const ParserByteCode& ParserBase::GetByteCode() const
{
return m_vRPN;
}


string_type ParserBase::GetVersion(EParserVersionInfo eInfo) const
{
stringstream_type ss;

ss << ParserVersion;

if (eInfo == pviFULL)
{
ss << _T(" (") << ParserVersionDate;
ss << std::dec << _T("; ") << sizeof(void*) * 8 << _T("BIT");

#ifdef _DEBUG
ss << _T("; DEBUG");
#else 
ss << _T("; RELEASE");
#endif

#ifdef _UNICODE
ss << _T("; UNICODE");
#else
#ifdef _MBCS
ss << _T("; MBCS");
#else
ss << _T("; ASCII");
#endif
#endif

#ifdef MUP_USE_OPENMP
ss << _T("; OPENMP");
#endif

ss << _T(")");
}

return ss.str();
}


void ParserBase::AddValIdent(identfun_type a_pCallback)
{
m_pTokenReader->AddValIdent(a_pCallback);
}


void ParserBase::SetVarFactory(facfun_type a_pFactory, void* pUserData)
{
m_pTokenReader->SetVarCreator(a_pFactory, pUserData);
}


void ParserBase::AddCallback(
const string_type& a_strName,
const ParserCallback& a_Callback,
funmap_type& a_Storage,
const char_type* a_szCharSet)
{
if (!a_Callback.IsValid())
Error(ecINVALID_FUN_PTR);

const funmap_type* pFunMap = &a_Storage;

if (pFunMap != &m_FunDef && m_FunDef.find(a_strName) != m_FunDef.end())
Error(ecNAME_CONFLICT, -1, a_strName);

if (pFunMap != &m_PostOprtDef && m_PostOprtDef.find(a_strName) != m_PostOprtDef.end())
Error(ecNAME_CONFLICT, -1, a_strName);

if (pFunMap != &m_InfixOprtDef && pFunMap != &m_OprtDef && m_InfixOprtDef.find(a_strName) != m_InfixOprtDef.end())
Error(ecNAME_CONFLICT, -1, a_strName);

if (pFunMap != &m_InfixOprtDef && pFunMap != &m_OprtDef && m_OprtDef.find(a_strName) != m_OprtDef.end())
Error(ecNAME_CONFLICT, -1, a_strName);

CheckOprt(a_strName, a_Callback, a_szCharSet);
a_Storage[a_strName] = a_Callback;
ReInit();
}


void ParserBase::CheckOprt(const string_type& a_sName,
const ParserCallback& a_Callback,
const string_type& a_szCharSet) const
{
if (!a_sName.length() ||
(a_sName.find_first_not_of(a_szCharSet) != string_type::npos) ||
(a_sName[0] >= '0' && a_sName[0] <= '9'))
{
switch (a_Callback.GetCode())
{
case cmOPRT_POSTFIX: Error(ecINVALID_POSTFIX_IDENT, -1, a_sName); break;
case cmOPRT_INFIX:   Error(ecINVALID_INFIX_IDENT, -1, a_sName); break;
default:             Error(ecINVALID_NAME, -1, a_sName);
}
}
}



void ParserBase::CheckName(const string_type& a_sName, const string_type& a_szCharSet) const
{
if (!a_sName.length() ||
(a_sName.find_first_not_of(a_szCharSet) != string_type::npos) ||
(a_sName[0] >= '0' && a_sName[0] <= '9'))
{
Error(ecINVALID_NAME);
}
}


void ParserBase::SetExpr(const string_type& a_sExpr)
{
if (m_pTokenReader->GetArgSep() == std::use_facet<numpunct<char_type> >(s_locale).decimal_point())
Error(ecLOCALE);

if (a_sExpr.length() >= MaxLenExpression)
Error(ecEXPRESSION_TOO_LONG, 0, a_sExpr);

m_pTokenReader->SetFormula(a_sExpr + _T(" "));
ReInit();
}


const char_type** ParserBase::GetOprtDef() const
{
return (const char_type**)(&c_DefaultOprt[0]);
}


void ParserBase::DefineNameChars(const char_type* a_szCharset)
{
m_sNameChars = a_szCharset;
}


void ParserBase::DefineOprtChars(const char_type* a_szCharset)
{
m_sOprtChars = a_szCharset;
}


void ParserBase::DefineInfixOprtChars(const char_type* a_szCharset)
{
m_sInfixOprtChars = a_szCharset;
}


const char_type* ParserBase::ValidNameChars() const
{
MUP_ASSERT(m_sNameChars.size());
return m_sNameChars.c_str();
}


const char_type* ParserBase::ValidOprtChars() const
{
MUP_ASSERT(m_sOprtChars.size());
return m_sOprtChars.c_str();
}


const char_type* ParserBase::ValidInfixOprtChars() const
{
MUP_ASSERT(m_sInfixOprtChars.size());
return m_sInfixOprtChars.c_str();
}


void ParserBase::DefinePostfixOprt(const string_type& a_sName, fun_type1 a_pFun, bool a_bAllowOpt)
{
if (a_sName.length() > MaxLenIdentifier)
Error(ecIDENTIFIER_TOO_LONG);

AddCallback(a_sName, ParserCallback(a_pFun, a_bAllowOpt, prPOSTFIX, cmOPRT_POSTFIX), m_PostOprtDef, ValidOprtChars());
}


void ParserBase::Init()
{
InitCharSets();
InitFun();
InitConst();
InitOprt();
}


void ParserBase::DefineInfixOprt(const string_type& a_sName, fun_type1 a_pFun, int a_iPrec, bool a_bAllowOpt)
{
if (a_sName.length() > MaxLenIdentifier)
Error(ecIDENTIFIER_TOO_LONG);

AddCallback(a_sName, ParserCallback(a_pFun, a_bAllowOpt, a_iPrec, cmOPRT_INFIX), m_InfixOprtDef, ValidInfixOprtChars());
}



void ParserBase::DefineOprt(const string_type& a_sName, fun_type2 a_pFun, unsigned a_iPrec, EOprtAssociativity a_eAssociativity, bool a_bAllowOpt)
{
if (a_sName.length() > MaxLenIdentifier)
Error(ecIDENTIFIER_TOO_LONG);

for (int i = 0; m_bBuiltInOp && i < cmENDIF; ++i)
{
if (a_sName == string_type(c_DefaultOprt[i]))
{
Error(ecBUILTIN_OVERLOAD, -1, a_sName);
}
}

AddCallback(a_sName, ParserCallback(a_pFun, a_bAllowOpt, a_iPrec, a_eAssociativity), m_OprtDef, ValidOprtChars());
}


void ParserBase::DefineStrConst(const string_type& a_strName, const string_type& a_strVal)
{
if (m_StrVarDef.find(a_strName) != m_StrVarDef.end())
Error(ecNAME_CONFLICT);

CheckName(a_strName, ValidNameChars());

m_vStringVarBuf.push_back(a_strVal);                
m_StrVarDef[a_strName] = m_vStringVarBuf.size() - 1;  

ReInit();
}


void ParserBase::DefineVar(const string_type& a_sName, value_type* a_pVar)
{
if (a_pVar == 0)
Error(ecINVALID_VAR_PTR);

if (a_sName.length() > MaxLenIdentifier)
Error(ecIDENTIFIER_TOO_LONG);

if (m_ConstDef.find(a_sName) != m_ConstDef.end())
Error(ecNAME_CONFLICT);

CheckName(a_sName, ValidNameChars());
m_VarDef[a_sName] = a_pVar;
ReInit();
}


void ParserBase::DefineConst(const string_type& a_sName, value_type a_fVal)
{
if (a_sName.length() > MaxLenIdentifier)
Error(ecIDENTIFIER_TOO_LONG);

CheckName(a_sName, ValidNameChars());
m_ConstDef[a_sName] = a_fVal;
ReInit();
}


int ParserBase::GetOprtPrecedence(const token_type& a_Tok) const
{
switch (a_Tok.GetCode())
{
case cmEND:      return -5;
case cmARG_SEP:  return -4;
case cmASSIGN:   return -1;
case cmELSE:
case cmIF:       return  0;
case cmLAND:     return  prLAND;
case cmLOR:      return  prLOR;
case cmLT:
case cmGT:
case cmLE:
case cmGE:
case cmNEQ:
case cmEQ:       return  prCMP;
case cmADD:
case cmSUB:      return  prADD_SUB;
case cmMUL:
case cmDIV:      return  prMUL_DIV;
case cmPOW:      return  prPOW;

case cmOPRT_INFIX:
case cmOPRT_BIN: return a_Tok.GetPri();
default:  
throw exception_type(ecINTERNAL_ERROR, 5, _T(""));
}
}


EOprtAssociativity ParserBase::GetOprtAssociativity(const token_type& a_Tok) const
{
switch (a_Tok.GetCode())
{
case cmASSIGN:
case cmLAND:
case cmLOR:
case cmLT:
case cmGT:
case cmLE:
case cmGE:
case cmNEQ:
case cmEQ:
case cmADD:
case cmSUB:
case cmMUL:
case cmDIV:      return oaLEFT;
case cmPOW:      return oaRIGHT;
case cmOPRT_BIN: return a_Tok.GetAssociativity();
default:         return oaNONE;
}
}


const varmap_type& ParserBase::GetUsedVar() const
{
try
{
m_pTokenReader->IgnoreUndefVar(true);
CreateRPN(); 
m_pParseFormula = &ParserBase::ParseString;
m_pTokenReader->IgnoreUndefVar(false);
}
catch (exception_type& )
{
m_pParseFormula = &ParserBase::ParseString;
m_pTokenReader->IgnoreUndefVar(false);
throw;
}

return m_pTokenReader->GetUsedVar();
}


const varmap_type& ParserBase::GetVar() const
{
return m_VarDef;
}


const valmap_type& ParserBase::GetConst() const
{
return m_ConstDef;
}


const funmap_type& ParserBase::GetFunDef() const
{
return m_FunDef;
}


const string_type& ParserBase::GetExpr() const
{
return m_pTokenReader->GetExpr();
}


ParserBase::token_type ParserBase::ApplyStrFunc(
const token_type& a_FunTok,
const std::vector<token_type>& a_vArg) const
{
if (a_vArg.back().GetCode() != cmSTRING)
Error(ecSTRING_EXPECTED, m_pTokenReader->GetPos(), a_FunTok.GetAsString());

token_type  valTok;
generic_callable_type pFunc = a_FunTok.GetFuncAddr();
MUP_ASSERT(pFunc);

try
{
switch (a_FunTok.GetArgCount())
{
case 0: valTok.SetVal(1); a_vArg[0].GetAsString();  break;
case 1: valTok.SetVal(1); a_vArg[1].GetAsString();  a_vArg[0].GetVal();  break;
case 2: valTok.SetVal(1); a_vArg[2].GetAsString();  a_vArg[1].GetVal();  a_vArg[0].GetVal();  break;
case 3: valTok.SetVal(1); a_vArg[3].GetAsString();  a_vArg[2].GetVal();  a_vArg[1].GetVal();  a_vArg[0].GetVal();  break;
case 4: valTok.SetVal(1); a_vArg[4].GetAsString();  a_vArg[3].GetVal();  a_vArg[2].GetVal();  a_vArg[1].GetVal();  a_vArg[0].GetVal();  break;
case 5: valTok.SetVal(1); a_vArg[5].GetAsString();  a_vArg[4].GetVal();  a_vArg[3].GetVal();  a_vArg[2].GetVal();  a_vArg[1].GetVal(); a_vArg[0].GetVal(); break;
default: Error(ecINTERNAL_ERROR);
}
}
catch (ParserError&)
{
Error(ecVAL_EXPECTED, m_pTokenReader->GetPos(), a_FunTok.GetAsString());
}

m_vRPN.AddStrFun(pFunc, a_FunTok.GetArgCount(), a_vArg.back().GetIdx());

return valTok;
}


void ParserBase::ApplyFunc(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal, int a_iArgCount) const
{
MUP_ASSERT(m_pTokenReader.get());

if (a_stOpt.empty() || a_stOpt.top().GetFuncAddr() == 0)
return;

token_type funTok = a_stOpt.top();
a_stOpt.pop();
MUP_ASSERT(funTok.GetFuncAddr() != nullptr);

int iArgCount = (funTok.GetCode() == cmOPRT_BIN) ? funTok.GetArgCount() : a_iArgCount;

int iArgRequired = funTok.GetArgCount() + ((funTok.GetType() == tpSTR) ? 1 : 0);

int iArgNumerical = iArgCount - ((funTok.GetType() == tpSTR) ? 1 : 0);

if (funTok.GetCode() == cmFUNC_STR && iArgCount - iArgNumerical > 1)
Error(ecINTERNAL_ERROR);

if (funTok.GetArgCount() >= 0 && iArgCount > iArgRequired)
Error(ecTOO_MANY_PARAMS, m_pTokenReader->GetPos() - 1, funTok.GetAsString());

if (funTok.GetCode() != cmOPRT_BIN && iArgCount < iArgRequired)
Error(ecTOO_FEW_PARAMS, m_pTokenReader->GetPos() - 1, funTok.GetAsString());

if (funTok.GetCode() == cmFUNC_STR && iArgCount > iArgRequired)
Error(ecTOO_MANY_PARAMS, m_pTokenReader->GetPos() - 1, funTok.GetAsString());

std::vector<token_type> stArg;
for (int i = 0; i < iArgNumerical; ++i)
{
if (a_stVal.empty())
Error(ecINTERNAL_ERROR, m_pTokenReader->GetPos(), funTok.GetAsString());

stArg.push_back(a_stVal.top());
a_stVal.pop();

if (stArg.back().GetType() == tpSTR && funTok.GetType() != tpSTR)
Error(ecVAL_EXPECTED, m_pTokenReader->GetPos(), funTok.GetAsString());
}

switch (funTok.GetCode())
{
case  cmFUNC_STR:
if (a_stVal.empty())
Error(ecINTERNAL_ERROR, m_pTokenReader->GetPos(), funTok.GetAsString());

stArg.push_back(a_stVal.top());
a_stVal.pop();

if (stArg.back().GetType() == tpSTR && funTok.GetType() != tpSTR)
Error(ecVAL_EXPECTED, m_pTokenReader->GetPos(), funTok.GetAsString());

ApplyStrFunc(funTok, stArg);
break;

case  cmFUNC_BULK:
m_vRPN.AddBulkFun(funTok.GetFuncAddr(), (int)stArg.size());
break;

case  cmOPRT_BIN:
case  cmOPRT_POSTFIX:
case  cmOPRT_INFIX:
case  cmFUNC:
if (funTok.GetArgCount() == -1 && iArgCount == 0)
Error(ecTOO_FEW_PARAMS, m_pTokenReader->GetPos(), funTok.GetAsString());

m_vRPN.AddFun(funTok.GetFuncAddr(), (funTok.GetArgCount() == -1) ? -iArgNumerical : iArgNumerical, funTok.IsOptimizable());
break;
default:
break;
}

token_type token;
token.SetVal(1);
a_stVal.push(token);
}

void ParserBase::ApplyIfElse(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal) const
{
while (a_stOpt.size() && a_stOpt.top().GetCode() == cmELSE)
{
MUP_ASSERT(!a_stOpt.empty())
token_type opElse = a_stOpt.top();
a_stOpt.pop();

MUP_ASSERT(!a_stVal.empty());
token_type vVal2 = a_stVal.top();
if (vVal2.GetType() != tpDBL)
Error(ecUNEXPECTED_STR, m_pTokenReader->GetPos());

a_stVal.pop();

MUP_ASSERT(!a_stVal.empty());
token_type vVal1 = a_stVal.top();
if (vVal1.GetType() != tpDBL)
Error(ecUNEXPECTED_STR, m_pTokenReader->GetPos());

a_stVal.pop();

MUP_ASSERT(!a_stVal.empty());
token_type vExpr = a_stVal.top();
a_stVal.pop();

a_stVal.push((vExpr.GetVal() != 0) ? vVal1 : vVal2);

token_type opIf = a_stOpt.top();
a_stOpt.pop();

MUP_ASSERT(opElse.GetCode() == cmELSE);

if (opIf.GetCode() != cmIF)
Error(ecMISPLACED_COLON, m_pTokenReader->GetPos());

m_vRPN.AddIfElse(cmENDIF);
} 
}


void ParserBase::ApplyBinOprt(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal) const
{
if (a_stOpt.top().GetCode() == cmOPRT_BIN)
{
ApplyFunc(a_stOpt, a_stVal, 2);
}
else
{
if (a_stVal.size() < 2)
Error(ecINTERNAL_ERROR, m_pTokenReader->GetPos(), _T("ApplyBinOprt: not enough values in value stack!"));

token_type valTok1 = a_stVal.top();
a_stVal.pop();

token_type valTok2 = a_stVal.top();
a_stVal.pop();

token_type optTok = a_stOpt.top();
a_stOpt.pop();

token_type resTok;

if (valTok1.GetType() != valTok2.GetType() ||
(valTok1.GetType() == tpSTR && valTok2.GetType() == tpSTR))
Error(ecOPRT_TYPE_CONFLICT, m_pTokenReader->GetPos(), optTok.GetAsString());

if (optTok.GetCode() == cmASSIGN)
{
if (valTok2.GetCode() != cmVAR)
Error(ecUNEXPECTED_OPERATOR, -1, _T("="));

m_vRPN.AddAssignOp(valTok2.GetVar());
}
else
m_vRPN.AddOp(optTok.GetCode());

resTok.SetVal(1);
a_stVal.push(resTok);
}
}


void ParserBase::ApplyRemainingOprt(std::stack<token_type>& stOpt, std::stack<token_type>& stVal) const
{
while (stOpt.size() &&
stOpt.top().GetCode() != cmBO &&
stOpt.top().GetCode() != cmIF)
{
token_type tok = stOpt.top();
switch (tok.GetCode())
{
case cmOPRT_INFIX:
case cmOPRT_BIN:
case cmLE:
case cmGE:
case cmNEQ:
case cmEQ:
case cmLT:
case cmGT:
case cmADD:
case cmSUB:
case cmMUL:
case cmDIV:
case cmPOW:
case cmLAND:
case cmLOR:
case cmASSIGN:
if (stOpt.top().GetCode() == cmOPRT_INFIX)
ApplyFunc(stOpt, stVal, 1);
else
ApplyBinOprt(stOpt, stVal);
break;

case cmELSE:
ApplyIfElse(stOpt, stVal);
break;

default:
Error(ecINTERNAL_ERROR);
}
}
}


value_type ParserBase::ParseCmdCode() const
{
return ParseCmdCodeBulk(0, 0);
}

value_type ParserBase::ParseCmdCodeShort() const
{
const SToken *const tok = m_vRPN.GetBase();
value_type buf;

switch (tok->Cmd)
{
case cmVAL:		
return tok->Val.data2;

case cmVAR:		
return *tok->Val.ptr;

case cmVARMUL:	
return *tok->Val.ptr * tok->Val.data + tok->Val.data2;

case cmVARPOW2: 
buf = *(tok->Val.ptr);
return buf * buf;

case  cmVARPOW3: 				
buf = *(tok->Val.ptr);
return buf * buf * buf;

case  cmVARPOW4: 				
buf = *(tok->Val.ptr);
return buf * buf * buf * buf;

case cmFUNC:
return tok->Fun.cb.call_fun<0>();

case cmFUNC_STR:
return tok->Fun.cb.call_strfun<1>(m_vStringBuf[0].c_str());

default:
throw ParserError(ecINTERNAL_ERROR);
}
}


value_type ParserBase::ParseCmdCodeBulk(int nOffset, int nThreadID) const
{
assert(nThreadID <= s_MaxNumOpenMPThreads);

value_type *stack = ((nOffset == 0) && (nThreadID == 0)) ? &m_vStackBuffer[0] : &m_vStackBuffer[nThreadID * (m_vStackBuffer.size() / s_MaxNumOpenMPThreads)];
value_type buf;
int sidx(0);
for (const SToken* pTok = m_vRPN.GetBase(); pTok->Cmd != cmEND; ++pTok)
{
switch (pTok->Cmd)
{
case  cmLE:   --sidx; stack[sidx] = stack[sidx] <= stack[sidx + 1]; continue;
case  cmGE:   --sidx; stack[sidx] = stack[sidx] >= stack[sidx + 1]; continue;
case  cmNEQ:  --sidx; stack[sidx] = stack[sidx] != stack[sidx + 1]; continue;
case  cmEQ:   --sidx; stack[sidx] = stack[sidx] == stack[sidx + 1]; continue;
case  cmLT:   --sidx; stack[sidx] = stack[sidx] < stack[sidx + 1];  continue;
case  cmGT:   --sidx; stack[sidx] = stack[sidx] > stack[sidx + 1];  continue;
case  cmADD:  --sidx; stack[sidx] += stack[1 + sidx]; continue;
case  cmSUB:  --sidx; stack[sidx] -= stack[1 + sidx]; continue;
case  cmMUL:  --sidx; stack[sidx] *= stack[1 + sidx]; continue;
case  cmDIV:  --sidx;
stack[sidx] /= stack[1 + sidx];
continue;

case  cmPOW:
--sidx; stack[sidx] = MathImpl<value_type>::Pow(stack[sidx], stack[1 + sidx]);
continue;

case  cmLAND: --sidx; stack[sidx] = stack[sidx] && stack[sidx + 1]; continue;
case  cmLOR:  --sidx; stack[sidx] = stack[sidx] || stack[sidx + 1]; continue;

case  cmASSIGN:
--sidx; 
stack[sidx] = *(pTok->Oprt.ptr + nOffset) = stack[sidx + 1]; 
continue;

case  cmIF:
if (stack[sidx--] == 0)
{
MUP_ASSERT(sidx >= 0);
pTok += pTok->Oprt.offset;
}
continue;

case  cmELSE:
pTok += pTok->Oprt.offset;
continue;

case  cmENDIF:
continue;

case  cmVAR:    stack[++sidx] = *(pTok->Val.ptr + nOffset);  continue;
case  cmVAL:    stack[++sidx] = pTok->Val.data2;  continue;

case  cmVARPOW2: buf = *(pTok->Val.ptr + nOffset);
stack[++sidx] = buf * buf;
continue;

case  cmVARPOW3: buf = *(pTok->Val.ptr + nOffset);
stack[++sidx] = buf * buf * buf;
continue;

case  cmVARPOW4: buf = *(pTok->Val.ptr + nOffset);
stack[++sidx] = buf * buf * buf * buf;
continue;

case  cmVARMUL:  
stack[++sidx] = *(pTok->Val.ptr + nOffset) * pTok->Val.data + pTok->Val.data2;
continue;

case  cmFUNC:
{
int iArgCount = pTok->Fun.argc;

switch (iArgCount)
{
case 0: sidx += 1; stack[sidx] = pTok->Fun.cb.call_fun<0 >(); continue;
case 1:            stack[sidx] = pTok->Fun.cb.call_fun<1 >(stack[sidx]);   continue;
case 2: sidx -= 1; stack[sidx] = pTok->Fun.cb.call_fun<2 >(stack[sidx], stack[sidx + 1]); continue;
case 3: sidx -= 2; stack[sidx] = pTok->Fun.cb.call_fun<3 >(stack[sidx], stack[sidx + 1], stack[sidx + 2]); continue;
case 4: sidx -= 3; stack[sidx] = pTok->Fun.cb.call_fun<4 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3]); continue;
case 5: sidx -= 4; stack[sidx] = pTok->Fun.cb.call_fun<5 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4]); continue;
case 6: sidx -= 5; stack[sidx] = pTok->Fun.cb.call_fun<6 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5]); continue;
case 7: sidx -= 6; stack[sidx] = pTok->Fun.cb.call_fun<7 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6]); continue;
case 8: sidx -= 7; stack[sidx] = pTok->Fun.cb.call_fun<8 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7]); continue;
case 9: sidx -= 8; stack[sidx] = pTok->Fun.cb.call_fun<9 >(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7], stack[sidx + 8]); continue;
case 10:sidx -= 9; stack[sidx] = pTok->Fun.cb.call_fun<10>(stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7], stack[sidx + 8], stack[sidx + 9]); continue;
default:
if (iArgCount > 0)
Error(ecINTERNAL_ERROR, -1);

sidx -= -iArgCount - 1;

if (sidx <= 0)
Error(ecINTERNAL_ERROR, -1);

stack[sidx] = pTok->Fun.cb.call_multfun(&stack[sidx], -iArgCount);
continue;
}
}

case  cmFUNC_STR:
{
sidx -= pTok->Fun.argc - 1;

int iIdxStack = pTok->Fun.idx;
if (iIdxStack < 0 || iIdxStack >= (int)m_vStringBuf.size())
Error(ecINTERNAL_ERROR, m_pTokenReader->GetPos());

switch (pTok->Fun.argc)  
{
case 0: stack[sidx] = pTok->Fun.cb.call_strfun<1>(m_vStringBuf[iIdxStack].c_str()); continue;
case 1: stack[sidx] = pTok->Fun.cb.call_strfun<2>(m_vStringBuf[iIdxStack].c_str(), stack[sidx]); continue;
case 2: stack[sidx] = pTok->Fun.cb.call_strfun<3>(m_vStringBuf[iIdxStack].c_str(), stack[sidx], stack[sidx + 1]); continue;
case 3: stack[sidx] = pTok->Fun.cb.call_strfun<4>(m_vStringBuf[iIdxStack].c_str(), stack[sidx], stack[sidx + 1], stack[sidx + 2]); continue;
case 4: stack[sidx] = pTok->Fun.cb.call_strfun<5>(m_vStringBuf[iIdxStack].c_str(), stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3]); continue;
case 5: stack[sidx] = pTok->Fun.cb.call_strfun<6>(m_vStringBuf[iIdxStack].c_str(), stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4]); continue;
}

continue;
}

case  cmFUNC_BULK:
{
int iArgCount = pTok->Fun.argc;

switch (iArgCount)
{
case 0: sidx += 1; stack[sidx] = pTok->Fun.cb.call_bulkfun<0 >(nOffset, nThreadID); continue;
case 1:            stack[sidx] = pTok->Fun.cb.call_bulkfun<1 >(nOffset, nThreadID, stack[sidx]); continue;
case 2: sidx -= 1; stack[sidx] = pTok->Fun.cb.call_bulkfun<2 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1]); continue;
case 3: sidx -= 2; stack[sidx] = pTok->Fun.cb.call_bulkfun<3 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2]); continue;
case 4: sidx -= 3; stack[sidx] = pTok->Fun.cb.call_bulkfun<4 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3]); continue;
case 5: sidx -= 4; stack[sidx] = pTok->Fun.cb.call_bulkfun<5 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4]); continue;
case 6: sidx -= 5; stack[sidx] = pTok->Fun.cb.call_bulkfun<6 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5]); continue;
case 7: sidx -= 6; stack[sidx] = pTok->Fun.cb.call_bulkfun<7 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6]); continue;
case 8: sidx -= 7; stack[sidx] = pTok->Fun.cb.call_bulkfun<8 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7]); continue;
case 9: sidx -= 8; stack[sidx] = pTok->Fun.cb.call_bulkfun<9 >(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7], stack[sidx + 8]); continue;
case 10:sidx -= 9; stack[sidx] = pTok->Fun.cb.call_bulkfun<10>(nOffset, nThreadID, stack[sidx], stack[sidx + 1], stack[sidx + 2], stack[sidx + 3], stack[sidx + 4], stack[sidx + 5], stack[sidx + 6], stack[sidx + 7], stack[sidx + 8], stack[sidx + 9]); continue;
default:
throw exception_type(ecINTERNAL_ERROR, 2, _T(""));
}
}

default:
throw exception_type(ecINTERNAL_ERROR, 3, _T(""));
} 
} 

return stack[m_nFinalResultIdx];
}

void ParserBase::CreateRPN() const
{
if (!m_pTokenReader->GetExpr().length())
Error(ecUNEXPECTED_EOF, 0);

std::stack<token_type> stOpt, stVal;
std::stack<int> stArgCount;
token_type opta, opt;  
token_type val, tval;  
int ifElseCounter = 0;

ReInit();

stArgCount.push(1);

for (;;)
{
opt = m_pTokenReader->ReadNextToken();

switch (opt.GetCode())
{
case cmSTRING:
if (stOpt.empty())
Error(ecSTR_RESULT, m_pTokenReader->GetPos(), opt.GetAsString());

opt.SetIdx((int)m_vStringBuf.size());      
stVal.push(opt);
m_vStringBuf.push_back(opt.GetAsString()); 
break;

case cmVAR:
stVal.push(opt);
m_vRPN.AddVar(static_cast<value_type*>(opt.GetVar()));
break;

case cmVAL:
stVal.push(opt);
m_vRPN.AddVal(opt.GetVal());
break;

case cmELSE:
if (stArgCount.empty())
Error(ecMISPLACED_COLON, m_pTokenReader->GetPos());

if (stArgCount.top() > 1)
Error(ecUNEXPECTED_ARG_SEP, m_pTokenReader->GetPos());

stArgCount.pop();

ifElseCounter--;
if (ifElseCounter < 0)
Error(ecMISPLACED_COLON, m_pTokenReader->GetPos());

ApplyRemainingOprt(stOpt, stVal);
m_vRPN.AddIfElse(cmELSE);
stOpt.push(opt);
break;

case cmARG_SEP:
if (!stOpt.empty() && stOpt.top().GetCode() == cmIF)
Error(ecUNEXPECTED_ARG_SEP, m_pTokenReader->GetPos());

if (stArgCount.empty())
Error(ecUNEXPECTED_ARG_SEP, m_pTokenReader->GetPos());

++stArgCount.top();

case cmEND:
ApplyRemainingOprt(stOpt, stVal);
break;

case cmBC:
{
if (opta.GetCode() == cmBO)
--stArgCount.top();

ApplyRemainingOprt(stOpt, stVal);

if (stOpt.size() && stOpt.top().GetCode() == cmBO)
{
MUP_ASSERT(stArgCount.size());
int iArgCount = stArgCount.top();
stArgCount.pop();

stOpt.pop(); 

if (iArgCount > 1 && (stOpt.size() == 0 ||
(stOpt.top().GetCode() != cmFUNC &&
stOpt.top().GetCode() != cmFUNC_BULK &&
stOpt.top().GetCode() != cmFUNC_STR)))
Error(ecUNEXPECTED_ARG, m_pTokenReader->GetPos());

if (stOpt.size() &&
stOpt.top().GetCode() != cmOPRT_INFIX &&
stOpt.top().GetCode() != cmOPRT_BIN &&
stOpt.top().GetFuncAddr() != 0)
{
ApplyFunc(stOpt, stVal, iArgCount);
}
}
} 
break;

case cmIF:
ifElseCounter++;
stArgCount.push(1);

case cmLAND:
case cmLOR:
case cmLT:
case cmGT:
case cmLE:
case cmGE:
case cmNEQ:
case cmEQ:
case cmADD:
case cmSUB:
case cmMUL:
case cmDIV:
case cmPOW:
case cmASSIGN:
case cmOPRT_BIN:

while (
stOpt.size() &&
stOpt.top().GetCode() != cmBO &&
stOpt.top().GetCode() != cmELSE &&
stOpt.top().GetCode() != cmIF)
{
int nPrec1 = GetOprtPrecedence(stOpt.top()),
nPrec2 = GetOprtPrecedence(opt);

if (stOpt.top().GetCode() == opt.GetCode())
{

EOprtAssociativity eOprtAsct = GetOprtAssociativity(opt);
if ((eOprtAsct == oaRIGHT && (nPrec1 <= nPrec2)) ||
(eOprtAsct == oaLEFT && (nPrec1 < nPrec2)))
{
break;
}
}
else if (nPrec1 < nPrec2)
{
break;
}

if (stOpt.top().GetCode() == cmOPRT_INFIX)
ApplyFunc(stOpt, stVal, 1);
else
ApplyBinOprt(stOpt, stVal);
} 

if (opt.GetCode() == cmIF)
m_vRPN.AddIfElse(opt.GetCode());

stOpt.push(opt);
break;

case cmBO:
stArgCount.push(1);
stOpt.push(opt);
break;

case cmOPRT_INFIX:
case cmFUNC:
case cmFUNC_BULK:
case cmFUNC_STR:
stOpt.push(opt);
break;

case cmOPRT_POSTFIX:
stOpt.push(opt);
ApplyFunc(stOpt, stVal, 1);  
break;

default:  Error(ecINTERNAL_ERROR, 3);
} 

opta = opt;

if (opt.GetCode() == cmEND)
{
m_vRPN.Finalize();
break;
}

if (ParserBase::g_DbgDumpStack)
{
StackDump(stVal, stOpt);
m_vRPN.AsciiDump();
}

} 

if (ParserBase::g_DbgDumpCmdCode)
m_vRPN.AsciiDump();

if (ifElseCounter > 0)
Error(ecMISSING_ELSE_CLAUSE);

MUP_ASSERT(stArgCount.size() == 1);
m_nFinalResultIdx = stArgCount.top();
if (m_nFinalResultIdx == 0)
Error(ecINTERNAL_ERROR, 9);

if (stVal.size() == 0)
Error(ecEMPTY_EXPRESSION);

while (stVal.size())
{
if (stVal.top().GetType() != tpDBL)
Error(ecSTR_RESULT);

stVal.pop();
}

m_vStackBuffer.resize(m_vRPN.GetMaxStackSize() * s_MaxNumOpenMPThreads);
}


value_type ParserBase::ParseString() const
{
try
{
CreateRPN();

if (m_vRPN.GetSize() == 2)
{
m_pParseFormula = &ParserBase::ParseCmdCodeShort;
m_vStackBuffer[1] = (this->*m_pParseFormula)();
return m_vStackBuffer[1];
}
else
{
m_pParseFormula = &ParserBase::ParseCmdCode;
return (this->*m_pParseFormula)();
}
}
catch (ParserError& exc)
{
exc.SetFormula(m_pTokenReader->GetExpr());
throw;
}
}


void  ParserBase::Error(EErrorCodes a_iErrc, int a_iPos, const string_type& a_sTok) const
{
throw exception_type(a_iErrc, a_sTok, m_pTokenReader->GetExpr(), a_iPos);
}


void ParserBase::ClearVar()
{
m_VarDef.clear();
ReInit();
}


void ParserBase::RemoveVar(const string_type& a_strVarName)
{
varmap_type::iterator item = m_VarDef.find(a_strVarName);
if (item != m_VarDef.end())
{
m_VarDef.erase(item);
ReInit();
}
}


void ParserBase::ClearFun()
{
m_FunDef.clear();
ReInit();
}


void ParserBase::ClearConst()
{
m_ConstDef.clear();
m_StrVarDef.clear();
ReInit();
}


void ParserBase::ClearPostfixOprt()
{
m_PostOprtDef.clear();
ReInit();
}


void ParserBase::ClearOprt()
{
m_OprtDef.clear();
ReInit();
}


void ParserBase::ClearInfixOprt()
{
m_InfixOprtDef.clear();
ReInit();
}


void ParserBase::EnableOptimizer(bool a_bIsOn)
{
m_vRPN.EnableOptimizer(a_bIsOn);
ReInit();
}


void ParserBase::EnableDebugDump(bool bDumpCmd, bool bDumpStack)
{
ParserBase::g_DbgDumpCmdCode = bDumpCmd;
ParserBase::g_DbgDumpStack = bDumpStack;
}


void ParserBase::EnableBuiltInOprt(bool a_bIsOn)
{
m_bBuiltInOp = a_bIsOn;
ReInit();
}


bool ParserBase::HasBuiltInOprt() const
{
return m_bBuiltInOp;
}


char_type ParserBase::GetArgSep() const
{
return m_pTokenReader->GetArgSep();
}


void ParserBase::SetArgSep(char_type cArgSep)
{
m_pTokenReader->SetArgSep(cArgSep);
}


void ParserBase::StackDump(const std::stack<token_type>& a_stVal, const std::stack<token_type>& a_stOprt) const
{
std::stack<token_type> stOprt(a_stOprt);
std::stack<token_type> stVal(a_stVal);

mu::console() << _T("\nValue stack:\n");
while (!stVal.empty())
{
token_type val = stVal.top();
stVal.pop();

if (val.GetType() == tpSTR)
mu::console() << _T(" \"") << val.GetAsString() << _T("\" ");
else
mu::console() << _T(" ") << val.GetVal() << _T(" ");
}
mu::console() << "\nOperator stack:\n";

while (!stOprt.empty())
{
if (stOprt.top().GetCode() <= cmASSIGN)
{
mu::console() << _T("OPRT_INTRNL \"")
<< ParserBase::c_DefaultOprt[stOprt.top().GetCode()]
<< _T("\" \n");
}
else
{
switch (stOprt.top().GetCode())
{
case cmVAR:   mu::console() << _T("VAR\n");  break;
case cmVAL:   mu::console() << _T("VAL\n");  break;
case cmFUNC:
mu::console()
<< _T("FUNC \"")
<< stOprt.top().GetAsString()
<< _T("\"\n");
break;

case cmFUNC_BULK:
mu::console()
<< _T("FUNC_BULK \"")
<< stOprt.top().GetAsString()
<< _T("\"\n");
break;

case cmOPRT_INFIX:
mu::console() << _T("OPRT_INFIX \"")
<< stOprt.top().GetAsString()
<< _T("\"\n");
break;

case cmOPRT_BIN:
mu::console() << _T("OPRT_BIN \"")
<< stOprt.top().GetAsString()
<< _T("\"\n");
break;

case cmFUNC_STR: mu::console() << _T("FUNC_STR\n");       break;
case cmEND:      mu::console() << _T("END\n");            break;
case cmUNKNOWN:  mu::console() << _T("UNKNOWN\n");        break;
case cmBO:       mu::console() << _T("BRACKET \"(\"\n");  break;
case cmBC:       mu::console() << _T("BRACKET \")\"\n");  break;
case cmIF:       mu::console() << _T("IF\n");  break;
case cmELSE:     mu::console() << _T("ELSE\n");  break;
case cmENDIF:    mu::console() << _T("ENDIF\n");  break;
default:         mu::console() << stOprt.top().GetCode() << _T(" ");  break;
}
}
stOprt.pop();
}

mu::console() << dec << endl;
}


value_type ParserBase::Eval() const
{
return (this->*m_pParseFormula)();
}


value_type* ParserBase::Eval(int& nStackSize) const
{
if (m_vRPN.GetSize() > 0)
{
ParseCmdCode();
}
else
{
ParseString();
}

nStackSize = m_nFinalResultIdx;

return &m_vStackBuffer[1];
}


int ParserBase::GetNumResults() const
{
return m_nFinalResultIdx;
}

void ParserBase::Eval(value_type* results, int nBulkSize)
{
CreateRPN();

int i = 0;

#ifdef MUP_USE_OPENMP
#ifdef DEBUG_OMP_STUFF
int* pThread = new int[nBulkSize];
int* pIdx = new int[nBulkSize];
#endif

int nMaxThreads = std::min(omp_get_max_threads(), s_MaxNumOpenMPThreads);
int nThreadID = 0;

#ifdef DEBUG_OMP_STUFF
int ct = 0;
#endif
omp_set_num_threads(nMaxThreads);

#pragma omp parallel for schedule(static, std::max(nBulkSize/nMaxThreads, 1)) private(nThreadID)
for (i = 0; i < nBulkSize; ++i)
{
nThreadID = omp_get_thread_num();
results[i] = ParseCmdCodeBulk(i, nThreadID);

#ifdef DEBUG_OMP_STUFF
#pragma omp critical
{
pThread[ct] = nThreadID;
pIdx[ct] = i;
ct++;
}
#endif
}

#ifdef DEBUG_OMP_STUFF
FILE* pFile = fopen("bulk_dbg.txt", "w");
for (i = 0; i < nBulkSize; ++i)
{
fprintf(pFile, "idx: %d  thread: %d \n", pIdx[i], pThread[i]);
}

delete[] pIdx;
delete[] pThread;

fclose(pFile);
#endif

#else
for (i = 0; i < nBulkSize; ++i)
{
results[i] = ParseCmdCodeBulk(i, 0);
}
#endif

}
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

