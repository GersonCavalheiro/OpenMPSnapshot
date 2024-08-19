

#ifndef MU_PARSER_ERROR_H
#define MU_PARSER_ERROR_H

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include "muParserDef.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4251)  
#endif




namespace mu
{

class ParserErrorMsg final
{
public:
static const ParserErrorMsg& Instance();
string_type operator[](unsigned a_iIdx) const;

private:
ParserErrorMsg& operator=(const ParserErrorMsg&) = delete;
ParserErrorMsg(const ParserErrorMsg&) = delete;
ParserErrorMsg();

~ParserErrorMsg() = default;

std::vector<string_type>  m_vErrMsg;  
};



class API_EXPORT_CXX ParserError
{
private:


void ReplaceSubString(string_type& strSource, const string_type& strFind, const string_type& strReplaceWith);
void Reset();

public:

ParserError();
explicit ParserError(EErrorCodes a_iErrc);
explicit ParserError(const string_type& sMsg);
ParserError(EErrorCodes a_iErrc, const string_type& sTok, const string_type& sFormula = string_type(), int a_iPos = -1);
ParserError(EErrorCodes a_iErrc, int a_iPos, const string_type& sTok);
ParserError(const char_type* a_szMsg, int a_iPos = -1, const string_type& sTok = string_type());
ParserError(const ParserError& a_Obj);

ParserError& operator=(const ParserError& a_Obj);
~ParserError();

void SetFormula(const string_type& a_strFormula);
const string_type& GetExpr() const;
const string_type& GetMsg() const;
int GetPos() const;
const string_type& GetToken() const;
EErrorCodes GetCode() const;

private:
string_type m_strMsg;     
string_type m_strFormula; 
string_type m_strTok;     
int m_iPos;               
EErrorCodes m_iErrc;      
const ParserErrorMsg& m_ErrMsg;
};
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

