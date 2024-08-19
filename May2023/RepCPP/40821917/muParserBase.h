

#ifndef MU_PARSER_BASE_H
#define MU_PARSER_BASE_H

#include <cmath>
#include <string>
#include <iostream>
#include <map>
#include <memory>
#include <locale>
#include <limits.h>

#include "muParserDef.h"
#include "muParserTokenReader.h"
#include "muParserBytecode.h"
#include "muParserError.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4251)  
#endif


namespace mu
{



class API_EXPORT_CXX ParserBase
{
friend class ParserTokenReader;

private:


typedef value_type(ParserBase::* ParseFunction)() const;


typedef std::vector<value_type> valbuf_type;


typedef std::vector<string_type> stringbuf_type;


typedef ParserTokenReader token_reader_type;


typedef ParserToken<value_type, string_type> token_type;


static const int s_MaxNumOpenMPThreads;

public:


typedef ParserError exception_type;

static void EnableDebugDump(bool bDumpCmd, bool bDumpStack);

ParserBase();
ParserBase(const ParserBase& a_Parser);
ParserBase& operator=(const ParserBase& a_Parser);

virtual ~ParserBase();

value_type Eval() const;
value_type* Eval(int& nStackSize) const;
void Eval(value_type* results, int nBulkSize);

int GetNumResults() const;

void SetExpr(const string_type& a_sExpr);
void SetVarFactory(facfun_type a_pFactory, void* pUserData = nullptr);

void SetDecSep(char_type cDecSep);
void SetThousandsSep(char_type cThousandsSep = 0);
void ResetLocale();

void EnableOptimizer(bool a_bIsOn = true);
void EnableBuiltInOprt(bool a_bIsOn = true);

bool HasBuiltInOprt() const;
void AddValIdent(identfun_type a_pCallback);


template<typename T>
void DefineFun(const string_type& a_strName, T a_pFun, bool a_bAllowOpt = true)
{
AddCallback(a_strName, ParserCallback(a_pFun, a_bAllowOpt), m_FunDef, ValidNameChars());
}


template<typename T>
void DefineFunUserData(const string_type& a_strName, T a_pFun, void* a_pUserData, bool a_bAllowOpt = true)
{
AddCallback(a_strName, ParserCallback(a_pFun, a_pUserData, a_bAllowOpt), m_FunDef, ValidNameChars());
}

void DefineOprt(const string_type& a_strName, fun_type2 a_pFun, unsigned a_iPri = 0, EOprtAssociativity a_eAssociativity = oaLEFT, bool a_bAllowOpt = false);
void DefineConst(const string_type& a_sName, value_type a_fVal);
void DefineStrConst(const string_type& a_sName, const string_type& a_strVal);
void DefineVar(const string_type& a_sName, value_type* a_fVar);
void DefinePostfixOprt(const string_type& a_strFun, fun_type1 a_pOprt, bool a_bAllowOpt = true);
void DefineInfixOprt(const string_type& a_strName, fun_type1 a_pOprt, int a_iPrec = prINFIX, bool a_bAllowOpt = true);

void ClearVar();
void ClearFun();
void ClearConst();
void ClearInfixOprt();
void ClearPostfixOprt();
void ClearOprt();

void RemoveVar(const string_type& a_strVarName);
const varmap_type& GetUsedVar() const;
const varmap_type& GetVar() const;
const valmap_type& GetConst() const;
const string_type& GetExpr() const;
const funmap_type& GetFunDef() const;
string_type GetVersion(EParserVersionInfo eInfo = pviFULL) const;
const ParserByteCode& GetByteCode() const;

const char_type** GetOprtDef() const;
void DefineNameChars(const char_type* a_szCharset);
void DefineOprtChars(const char_type* a_szCharset);
void DefineInfixOprtChars(const char_type* a_szCharset);

const char_type* ValidNameChars() const;
const char_type* ValidOprtChars() const;
const char_type* ValidInfixOprtChars() const;

void SetArgSep(char_type cArgSep);
char_type GetArgSep() const;

protected:

void Init();
void Error(EErrorCodes a_iErrc, int a_iPos = static_cast<int>(mu::string_type::npos), const string_type& a_strTok = string_type()) const;

virtual void InitCharSets() = 0;
virtual void InitFun() = 0;
virtual void InitConst() = 0;
virtual void InitOprt() = 0;

virtual void OnDetectVar(string_type* pExpr, int& nStart, int& nEnd);

static const char_type* c_DefaultOprt[];
static std::locale s_locale;  
static bool g_DbgDumpCmdCode;
static bool g_DbgDumpStack;


template<class TChar>
class change_dec_sep : public std::numpunct<TChar>
{
public:

explicit change_dec_sep(char_type cDecSep, char_type cThousandsSep = 0, int nGroup = 3)
:std::numpunct<TChar>()
,m_nGroup(nGroup)
,m_cDecPoint(cDecSep)
,m_cThousandsSep(cThousandsSep)
{}

protected:

char_type do_decimal_point() const override
{
return m_cDecPoint;
}

char_type do_thousands_sep() const override
{
return m_cThousandsSep;
}

std::string do_grouping() const override
{
return std::string(1, (char)(m_cThousandsSep > 0 ? m_nGroup : CHAR_MAX));
}

private:

int m_nGroup;
char_type m_cDecPoint;
char_type m_cThousandsSep;
};

private:

void Assign(const ParserBase& a_Parser);
void InitTokenReader();
void ReInit() const;

void AddCallback(const string_type& a_strName, const ParserCallback& a_Callback, funmap_type& a_Storage, const char_type* a_szCharSet);
void ApplyRemainingOprt(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal) const;
void ApplyBinOprt(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal) const;
void ApplyIfElse(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal) const;
void ApplyFunc(std::stack<token_type>& a_stOpt, std::stack<token_type>& a_stVal, int iArgCount) const;

token_type ApplyStrFunc(const token_type& a_FunTok, const std::vector<token_type>& a_vArg) const;

int GetOprtPrecedence(const token_type& a_Tok) const;
EOprtAssociativity GetOprtAssociativity(const token_type& a_Tok) const;

void CreateRPN() const;

value_type ParseString() const;
value_type ParseCmdCode() const;
value_type ParseCmdCodeShort() const;
value_type ParseCmdCodeBulk(int nOffset, int nThreadID) const;

void  CheckName(const string_type& a_strName, const string_type& a_CharSet) const;
void  CheckOprt(const string_type& a_sName, const ParserCallback& a_Callback, const string_type& a_szCharSet) const;

void StackDump(const std::stack<token_type >& a_stVal, const std::stack<token_type >& a_stOprt) const;


mutable ParseFunction  m_pParseFormula;
mutable ParserByteCode m_vRPN;        
mutable stringbuf_type  m_vStringBuf; 
stringbuf_type  m_vStringVarBuf;

std::unique_ptr<token_reader_type> m_pTokenReader; 

funmap_type  m_FunDef;         
funmap_type  m_PostOprtDef;    
funmap_type  m_InfixOprtDef;   
funmap_type  m_OprtDef;        
valmap_type  m_ConstDef;       
strmap_type  m_StrVarDef;      
varmap_type  m_VarDef;         

bool m_bBuiltInOp;             

string_type m_sNameChars;      
string_type m_sOprtChars;      
string_type m_sInfixOprtChars; 

mutable valbuf_type m_vStackBuffer; 
mutable int m_nFinalResultIdx;
};

} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif
