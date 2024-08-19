

#include <cstdio>
#include <cstring>
#include <map>
#include <stack>
#include <string>

#include "muParserTokenReader.h"
#include "muParserBase.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif




namespace mu
{

class ParserBase;


ParserTokenReader::ParserTokenReader(const ParserTokenReader& a_Reader)
{
Assign(a_Reader);
}



ParserTokenReader& ParserTokenReader::operator=(const ParserTokenReader& a_Reader)
{
if (&a_Reader != this)
Assign(a_Reader);

return *this;
}



void ParserTokenReader::Assign(const ParserTokenReader& a_Reader)
{
m_pParser = a_Reader.m_pParser;
m_strFormula = a_Reader.m_strFormula;
m_iPos = a_Reader.m_iPos;
m_iSynFlags = a_Reader.m_iSynFlags;

m_UsedVar = a_Reader.m_UsedVar;
m_pFunDef = a_Reader.m_pFunDef;
m_pConstDef = a_Reader.m_pConstDef;
m_pVarDef = a_Reader.m_pVarDef;
m_pStrVarDef = a_Reader.m_pStrVarDef;
m_pPostOprtDef = a_Reader.m_pPostOprtDef;
m_pInfixOprtDef = a_Reader.m_pInfixOprtDef;
m_pOprtDef = a_Reader.m_pOprtDef;
m_bIgnoreUndefVar = a_Reader.m_bIgnoreUndefVar;
m_vIdentFun = a_Reader.m_vIdentFun;
m_pFactory = a_Reader.m_pFactory;
m_pFactoryData = a_Reader.m_pFactoryData;
m_bracketStack = a_Reader.m_bracketStack;
m_cArgSep = a_Reader.m_cArgSep;
m_fZero = a_Reader.m_fZero;
m_lastTok = a_Reader.m_lastTok;
}



ParserTokenReader::ParserTokenReader(ParserBase* a_pParent)
:m_pParser(a_pParent)
, m_strFormula()
, m_iPos(0)
, m_iSynFlags(0)
, m_bIgnoreUndefVar(false)
, m_pFunDef(nullptr)
, m_pPostOprtDef(nullptr)
, m_pInfixOprtDef(nullptr)
, m_pOprtDef(nullptr)
, m_pConstDef(nullptr)
, m_pStrVarDef(nullptr)
, m_pVarDef(nullptr)
, m_pFactory(nullptr)
, m_pFactoryData(nullptr)
, m_vIdentFun()
, m_UsedVar()
, m_fZero(0)
, m_bracketStack()
, m_lastTok()
, m_cArgSep(',')
{
MUP_ASSERT(m_pParser != nullptr);
SetParent(m_pParser);
}



ParserTokenReader* ParserTokenReader::Clone(ParserBase* a_pParent) const
{
std::unique_ptr<ParserTokenReader> ptr(new ParserTokenReader(*this));
ptr->SetParent(a_pParent);
return ptr.release();
}


ParserTokenReader::token_type& ParserTokenReader::SaveBeforeReturn(const token_type& tok)
{
m_lastTok = tok;
return m_lastTok;
}


void ParserTokenReader::AddValIdent(identfun_type a_pCallback)
{
m_vIdentFun.push_front(a_pCallback);
}


void ParserTokenReader::SetVarCreator(facfun_type a_pFactory, void* pUserData)
{
m_pFactory = a_pFactory;
m_pFactoryData = pUserData;
}



int ParserTokenReader::GetPos() const
{
return m_iPos;
}



const string_type& ParserTokenReader::GetExpr() const
{
return m_strFormula;
}



varmap_type& ParserTokenReader::GetUsedVar()
{
return m_UsedVar;
}



void ParserTokenReader::SetFormula(const string_type& a_strFormula)
{
m_strFormula = a_strFormula;
ReInit();
}



void ParserTokenReader::IgnoreUndefVar(bool bIgnore)
{
m_bIgnoreUndefVar = bIgnore;
}



void ParserTokenReader::ReInit()
{
m_iPos = 0;
m_iSynFlags = sfSTART_OF_LINE;
m_bracketStack = std::stack<int>();
m_UsedVar.clear();
m_lastTok = token_type();
}



ParserTokenReader::token_type ParserTokenReader::ReadNextToken()
{
MUP_ASSERT(m_pParser != nullptr);

const char_type* szExpr = m_strFormula.c_str();
token_type tok;

while (szExpr[m_iPos] > 0 && szExpr[m_iPos] <= 0x20)
{
if (szExpr[m_iPos] >= 14 && szExpr[m_iPos] <= 31)
Error(ecINVALID_CHARACTERS_FOUND, m_iPos);

++m_iPos;
}

if (IsEOF(tok))
return SaveBeforeReturn(tok);

if (IsOprt(tok))
return SaveBeforeReturn(tok);

if (IsFunTok(tok))
return SaveBeforeReturn(tok);

if (IsBuiltIn(tok))
return SaveBeforeReturn(tok);

if (IsArgSep(tok))
return SaveBeforeReturn(tok);

if (IsValTok(tok))
return SaveBeforeReturn(tok);

if (IsVarTok(tok))
return SaveBeforeReturn(tok);

if (IsStrVarTok(tok))
return SaveBeforeReturn(tok);

if (IsString(tok))
return SaveBeforeReturn(tok);

if (IsInfixOpTok(tok))
return SaveBeforeReturn(tok);

if (IsPostOpTok(tok))
return SaveBeforeReturn(tok);

if ((m_bIgnoreUndefVar || m_pFactory) && IsUndefVarTok(tok))
return SaveBeforeReturn(tok);

string_type strTok;
auto iEnd = ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos);
if (iEnd != m_iPos)
Error(ecUNASSIGNABLE_TOKEN, m_iPos, strTok);

Error(ecUNASSIGNABLE_TOKEN, m_iPos, m_strFormula.substr(m_iPos));
return token_type(); 
}


void ParserTokenReader::SetParent(ParserBase* a_pParent)
{
m_pParser = a_pParent;
m_pFunDef = &a_pParent->m_FunDef;
m_pOprtDef = &a_pParent->m_OprtDef;
m_pInfixOprtDef = &a_pParent->m_InfixOprtDef;
m_pPostOprtDef = &a_pParent->m_PostOprtDef;
m_pVarDef = &a_pParent->m_VarDef;
m_pStrVarDef = &a_pParent->m_StrVarDef;
m_pConstDef = &a_pParent->m_ConstDef;
}



int ParserTokenReader::ExtractToken(const char_type* a_szCharSet, string_type& a_sTok, std::size_t a_iPos) const
{
auto iEnd = m_strFormula.find_first_not_of(a_szCharSet, a_iPos);

if (iEnd == string_type::npos)
iEnd = m_strFormula.length();

if (a_iPos != iEnd)
a_sTok = string_type(m_strFormula.begin() + a_iPos, m_strFormula.begin() + iEnd);

return static_cast<int>(iEnd);
}



int ParserTokenReader::ExtractOperatorToken(string_type& a_sTok, std::size_t a_iPos) const
{
auto iEnd = m_strFormula.find_first_not_of(m_pParser->ValidOprtChars(), a_iPos);
if (iEnd == string_type::npos)
iEnd = m_strFormula.length();

if (a_iPos != iEnd)
{
a_sTok = string_type(m_strFormula.begin() + a_iPos, m_strFormula.begin() + iEnd);
return static_cast<int>(iEnd);
}
else
{
return ExtractToken(_T("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), a_sTok, (std::size_t)a_iPos);
}
}



bool ParserTokenReader::IsBuiltIn(token_type& a_Tok)
{
const char_type** const pOprtDef = m_pParser->GetOprtDef(),
* const szFormula = m_strFormula.c_str();

for (int i = 0; pOprtDef[i]; i++)
{
std::size_t len(std::char_traits<char_type>::length(pOprtDef[i]));
if (string_type(pOprtDef[i]) == string_type(szFormula + m_iPos, szFormula + m_iPos + len))
{
switch (i)
{
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
if (i == cmASSIGN && m_iSynFlags & noASSIGN)
Error(ecUNEXPECTED_OPERATOR, m_iPos, pOprtDef[i]);

if (!m_pParser->HasBuiltInOprt()) continue;
if (m_iSynFlags & noOPT)
{
if (IsInfixOpTok(a_Tok))
return true;

Error(ecUNEXPECTED_OPERATOR, m_iPos, pOprtDef[i]);
}

m_iSynFlags = noBC | noOPT | noARG_SEP | noPOSTOP | noASSIGN | noIF | noELSE | noEND;
break;

case cmBO:
if (m_iSynFlags & noBO)
Error(ecUNEXPECTED_PARENS, m_iPos, pOprtDef[i]);

if (m_lastTok.GetCode() == cmFUNC)
m_iSynFlags = noOPT | noEND | noARG_SEP | noPOSTOP | noASSIGN | noIF | noELSE;
else
m_iSynFlags = noBC | noOPT | noEND | noARG_SEP | noPOSTOP | noASSIGN | noIF | noELSE;

m_bracketStack.push(cmBO);
break;

case cmBC:
if (m_iSynFlags & noBC)
Error(ecUNEXPECTED_PARENS, m_iPos, pOprtDef[i]);

m_iSynFlags = noBO | noVAR | noVAL | noFUN | noINFIXOP | noSTR | noASSIGN;

if (!m_bracketStack.empty())
m_bracketStack.pop();
else
Error(ecUNEXPECTED_PARENS, m_iPos, pOprtDef[i]);
break;

case cmELSE:
if (m_iSynFlags & noELSE)
Error(ecUNEXPECTED_CONDITIONAL, m_iPos, pOprtDef[i]);

m_iSynFlags = noBC | noPOSTOP | noEND | noOPT | noIF | noELSE | noSTR;
break;

case cmIF:
if (m_iSynFlags & noIF)
Error(ecUNEXPECTED_CONDITIONAL, m_iPos, pOprtDef[i]);

m_iSynFlags = noBC | noPOSTOP | noEND | noOPT | noIF | noELSE | noSTR;
break;

default:      
Error(ecINTERNAL_ERROR);
} 

m_iPos += (int)len;
a_Tok.Set((ECmdCode)i, pOprtDef[i]);
return true;
} 
} 

return false;
}


bool ParserTokenReader::IsArgSep(token_type& a_Tok)
{
const char_type* szFormula = m_strFormula.c_str();

if (szFormula[m_iPos] == m_cArgSep)
{
char_type szSep[2];
szSep[0] = m_cArgSep;
szSep[1] = 0;

if (m_iSynFlags & noARG_SEP)
Error(ecUNEXPECTED_ARG_SEP, m_iPos, szSep);

m_iSynFlags = noBC | noOPT | noEND | noARG_SEP | noPOSTOP | noASSIGN;
m_iPos++;
a_Tok.Set(cmARG_SEP, szSep);
return true;
}

return false;
}



bool ParserTokenReader::IsEOF(token_type& a_Tok)
{
const char_type* szFormula = m_strFormula.c_str();

if (!szFormula[m_iPos] )
{
if (m_iSynFlags & noEND)
Error(ecUNEXPECTED_EOF, m_iPos);

if (!m_bracketStack.empty())
Error(ecMISSING_PARENS, m_iPos, _T(")"));

m_iSynFlags = 0;
a_Tok.Set(cmEND);
return true;
}

return false;
}



bool ParserTokenReader::IsInfixOpTok(token_type& a_Tok)
{
string_type sTok;
auto iEnd = ExtractToken(m_pParser->ValidInfixOprtChars(), sTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

funmap_type::const_reverse_iterator it = m_pInfixOprtDef->rbegin();
for (; it != m_pInfixOprtDef->rend(); ++it)
{
if (sTok.find(it->first) != 0)
continue;

a_Tok.Set(it->second, it->first);
m_iPos += (int)it->first.length();

if (m_iSynFlags & noINFIXOP)
Error(ecUNEXPECTED_OPERATOR, m_iPos, a_Tok.GetAsString());

m_iSynFlags = noPOSTOP | noINFIXOP | noOPT | noBC | noSTR | noASSIGN | noARG_SEP;
return true;
}

return false;


}



bool ParserTokenReader::IsFunTok(token_type& a_Tok)
{
string_type strTok;
auto iEnd = ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

funmap_type::const_iterator item = m_pFunDef->find(strTok);
if (item == m_pFunDef->end())
return false;

const char_type* szFormula = m_strFormula.c_str();
if (szFormula[iEnd] != '(')
return false;

a_Tok.Set(item->second, strTok);

m_iPos = (int)iEnd;
if (m_iSynFlags & noFUN)
Error(ecUNEXPECTED_FUN, m_iPos - (int)a_Tok.GetAsString().length(), a_Tok.GetAsString());

m_iSynFlags = noANY ^ noBO;
return true;
}



bool ParserTokenReader::IsOprt(token_type& a_Tok)
{
const char_type* const szExpr = m_strFormula.c_str();
string_type strTok;

auto iEnd = ExtractOperatorToken(strTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

const char_type** const pOprtDef = m_pParser->GetOprtDef();
for (int i = 0; m_pParser->HasBuiltInOprt() && pOprtDef[i]; ++i)
{
if (string_type(pOprtDef[i]) == strTok)
return false;
}

funmap_type::const_reverse_iterator it = m_pOprtDef->rbegin();
for (; it != m_pOprtDef->rend(); ++it)
{
const string_type& sID = it->first;
if (sID == string_type(szExpr + m_iPos, szExpr + m_iPos + sID.length()))
{
a_Tok.Set(it->second, strTok);

if (m_iSynFlags & noOPT)
{
if (IsInfixOpTok(a_Tok))
return true;
else
{
return false;
}

}

m_iPos += (int)sID.length();
m_iSynFlags = noBC | noOPT | noARG_SEP | noPOSTOP | noEND | noASSIGN;
return true;
}
}

return false;
}



bool ParserTokenReader::IsPostOpTok(token_type& a_Tok)
{
if (m_iSynFlags & noPOSTOP)
return false;


string_type sTok;
auto iEnd = ExtractToken(m_pParser->ValidOprtChars(), sTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

funmap_type::const_reverse_iterator it = m_pPostOprtDef->rbegin();
for (; it != m_pPostOprtDef->rend(); ++it)
{
if (sTok.find(it->first) != 0)
continue;

a_Tok.Set(it->second, sTok);
m_iPos += (int)it->first.length();

m_iSynFlags = noVAL | noVAR | noFUN | noBO | noPOSTOP | noSTR | noASSIGN;
return true;
}

return false;
}



bool ParserTokenReader::IsValTok(token_type& a_Tok)
{
MUP_ASSERT(m_pConstDef != nullptr);
MUP_ASSERT(m_pParser != nullptr);

string_type strTok;
value_type fVal(0);

auto iEnd = ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos);
if (iEnd != m_iPos)
{
valmap_type::const_iterator item = m_pConstDef->find(strTok);
if (item != m_pConstDef->end())
{
m_iPos = iEnd;
a_Tok.SetVal(item->second, strTok);

if (m_iSynFlags & noVAL)
Error(ecUNEXPECTED_VAL, m_iPos - (int)strTok.length(), strTok);

m_iSynFlags = noVAL | noVAR | noFUN | noBO | noINFIXOP | noSTR | noASSIGN;
return true;
}
}

std::list<identfun_type>::const_iterator item = m_vIdentFun.begin();
for (item = m_vIdentFun.begin(); item != m_vIdentFun.end(); ++item)
{
int iStart = m_iPos;
if ((*item)(m_strFormula.c_str() + m_iPos, &m_iPos, &fVal) == 1)
{
strTok.assign(m_strFormula.c_str(), iStart, (std::size_t)m_iPos - iStart);

if (m_iSynFlags & noVAL)
Error(ecUNEXPECTED_VAL, m_iPos - (int)strTok.length(), strTok);

a_Tok.SetVal(fVal, strTok);
m_iSynFlags = noVAL | noVAR | noFUN | noBO | noINFIXOP | noSTR | noASSIGN;
return true;
}
}

return false;
}



bool ParserTokenReader::IsVarTok(token_type& a_Tok)
{
if (m_pVarDef->empty())
return false;

string_type strTok;
auto iEnd = ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

varmap_type::const_iterator item = m_pVarDef->find(strTok);
if (item == m_pVarDef->end())
return false;

if (m_iSynFlags & noVAR)
Error(ecUNEXPECTED_VAR, m_iPos, strTok);

m_pParser->OnDetectVar(&m_strFormula, m_iPos, iEnd);

m_iPos = iEnd;
a_Tok.SetVar(item->second, strTok);
m_UsedVar[item->first] = item->second;  

m_iSynFlags = noVAL | noVAR | noFUN | noBO | noINFIXOP | noSTR;

return true;
}


bool ParserTokenReader::IsStrVarTok(token_type& a_Tok)
{
if (!m_pStrVarDef || m_pStrVarDef->empty())
return false;

string_type strTok;
auto iEnd = ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos);
if (iEnd == m_iPos)
return false;

strmap_type::const_iterator item = m_pStrVarDef->find(strTok);
if (item == m_pStrVarDef->end())
return false;

if (m_iSynFlags & noSTR)
Error(ecUNEXPECTED_VAR, m_iPos, strTok);

m_iPos = iEnd;
if (!m_pParser->m_vStringVarBuf.size())
Error(ecINTERNAL_ERROR);

a_Tok.SetString(m_pParser->m_vStringVarBuf[item->second], m_pParser->m_vStringVarBuf.size());

m_iSynFlags = noANY ^ (noBC | noOPT | noEND | noARG_SEP);
return true;
}




bool ParserTokenReader::IsUndefVarTok(token_type& a_Tok)
{
string_type strTok;
auto iEnd(ExtractToken(m_pParser->ValidNameChars(), strTok, (std::size_t)m_iPos));
if (iEnd == m_iPos)
return false;

if (m_iSynFlags & noVAR)
{
Error(ecUNEXPECTED_VAR, m_iPos - (int)a_Tok.GetAsString().length(), strTok);
}

if (m_pFactory)
{
value_type* fVar = m_pFactory(strTok.c_str(), m_pFactoryData);
a_Tok.SetVar(fVar, strTok);

(*m_pVarDef)[strTok] = fVar;
m_UsedVar[strTok] = fVar;  
}
else
{
a_Tok.SetVar((value_type*)&m_fZero, strTok);
m_UsedVar[strTok] = 0;  
}

m_iPos = iEnd;

m_iSynFlags = noVAL | noVAR | noFUN | noBO | noPOSTOP | noINFIXOP | noSTR;
return true;
}




bool ParserTokenReader::IsString(token_type& a_Tok)
{
if (m_strFormula[m_iPos] != '"')
return false;

string_type strBuf(&m_strFormula[(std::size_t)m_iPos + 1]);
std::size_t iEnd(0), iSkip(0);

for (iEnd = (int)strBuf.find(_T('\"')); iEnd != 0 && iEnd != string_type::npos; iEnd = (int)strBuf.find(_T('\"'), iEnd))
{
if (strBuf[iEnd - 1] != '\\') break;
strBuf.replace(iEnd - 1, 2, _T("\""));
iSkip++;
}

if (iEnd == string_type::npos)
Error(ecUNTERMINATED_STRING, m_iPos, _T("\""));

string_type strTok(strBuf.begin(), strBuf.begin() + iEnd);

if (m_iSynFlags & noSTR)
Error(ecUNEXPECTED_STR, m_iPos, strTok);

m_pParser->m_vStringBuf.push_back(strTok); 
a_Tok.SetString(strTok, m_pParser->m_vStringBuf.size());

m_iPos += (int)strTok.length() + 2 + (int)iSkip;  
m_iSynFlags = noANY ^ (noARG_SEP | noBC | noOPT | noEND);

return true;
}



void  ParserTokenReader::Error(EErrorCodes a_iErrc,	int a_iPos,	const string_type& a_sTok) const
{
m_pParser->Error(a_iErrc, a_iPos, a_sTok);
}


void ParserTokenReader::SetArgSep(char_type cArgSep)
{
m_cArgSep = cArgSep;
}


char_type ParserTokenReader::GetArgSep() const
{
return m_cArgSep;
}
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
