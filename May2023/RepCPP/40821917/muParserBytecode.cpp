

#include "muParserBytecode.h"

#include <algorithm>
#include <string>
#include <stack>
#include <vector>
#include <iostream>

#include "muParserDef.h"
#include "muParserError.h"
#include "muParserToken.h"
#include "muParserTemplateMagic.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif


namespace mu
{

ParserByteCode::ParserByteCode()
:m_iStackPos(0)
, m_iMaxStackSize(0)
, m_vRPN()
, m_bEnableOptimizer(true)
{
m_vRPN.reserve(50);
}



ParserByteCode::ParserByteCode(const ParserByteCode& a_ByteCode)
{
Assign(a_ByteCode);
}



ParserByteCode& ParserByteCode::operator=(const ParserByteCode& a_ByteCode)
{
Assign(a_ByteCode);
return *this;
}


void ParserByteCode::EnableOptimizer(bool bStat)
{
m_bEnableOptimizer = bStat;
}



void ParserByteCode::Assign(const ParserByteCode& a_ByteCode)
{
if (this == &a_ByteCode)
return;

m_iStackPos = a_ByteCode.m_iStackPos;
m_vRPN = a_ByteCode.m_vRPN;
m_iMaxStackSize = a_ByteCode.m_iMaxStackSize;
m_bEnableOptimizer = a_ByteCode.m_bEnableOptimizer;
}



void ParserByteCode::AddVar(value_type* a_pVar)
{
++m_iStackPos;
m_iMaxStackSize = std::max(m_iMaxStackSize, (size_t)m_iStackPos);

SToken tok;
tok.Cmd = cmVAR;
tok.Val.ptr = a_pVar;
tok.Val.data = 1;
tok.Val.data2 = 0;
m_vRPN.push_back(tok);
}



void ParserByteCode::AddVal(value_type a_fVal)
{
++m_iStackPos;
m_iMaxStackSize = std::max(m_iMaxStackSize, (size_t)m_iStackPos);

SToken tok;
tok.Cmd = cmVAL;
tok.Val.ptr = nullptr;
tok.Val.data = 0;
tok.Val.data2 = a_fVal;
m_vRPN.push_back(tok);
}


void ParserByteCode::ConstantFolding(ECmdCode a_Oprt)
{
std::size_t sz = m_vRPN.size();
value_type& x = m_vRPN[sz - 2].Val.data2;
value_type& y = m_vRPN[sz - 1].Val.data2;

switch (a_Oprt)
{
case cmLAND: x = (int)x && (int)y; m_vRPN.pop_back(); break;
case cmLOR:  x = (int)x || (int)y; m_vRPN.pop_back(); break;
case cmLT:   x = x < y;  m_vRPN.pop_back();  break;
case cmGT:   x = x > y;  m_vRPN.pop_back();  break;
case cmLE:   x = x <= y; m_vRPN.pop_back();  break;
case cmGE:   x = x >= y; m_vRPN.pop_back();  break;
case cmNEQ:  x = x != y; m_vRPN.pop_back();  break;
case cmEQ:   x = x == y; m_vRPN.pop_back();  break;
case cmADD:  x = x + y;  m_vRPN.pop_back();  break;
case cmSUB:  x = x - y;  m_vRPN.pop_back();  break;
case cmMUL:  x = x * y;  m_vRPN.pop_back();  break;
case cmDIV:
x = x / y;
m_vRPN.pop_back();
break;

case cmPOW: x = MathImpl<value_type>::Pow(x, y);
m_vRPN.pop_back();
break;

default:
break;
} 
}



void ParserByteCode::AddOp(ECmdCode a_Oprt)
{
bool bOptimized = false;

if (m_bEnableOptimizer)
{
std::size_t sz = m_vRPN.size();

if (sz >= 2 && m_vRPN[sz - 2].Cmd == cmVAL && m_vRPN[sz - 1].Cmd == cmVAL)
{
ConstantFolding(a_Oprt);
bOptimized = true;
}
else
{
switch (a_Oprt)
{
case  cmPOW:
if (m_vRPN[sz - 2].Cmd == cmVAR && m_vRPN[sz - 1].Cmd == cmVAL)
{
if (m_vRPN[sz - 1].Val.data2 == 0)
{
m_vRPN[sz - 2].Cmd = cmVAL;
m_vRPN[sz - 2].Val.ptr = nullptr;
m_vRPN[sz - 2].Val.data = 0;
m_vRPN[sz - 2].Val.data2 = 1;
}
else if (m_vRPN[sz - 1].Val.data2 == 1)
m_vRPN[sz - 2].Cmd = cmVAR;
else if (m_vRPN[sz - 1].Val.data2 == 2)
m_vRPN[sz - 2].Cmd = cmVARPOW2;
else if (m_vRPN[sz - 1].Val.data2 == 3)
m_vRPN[sz - 2].Cmd = cmVARPOW3;
else if (m_vRPN[sz - 1].Val.data2 == 4)
m_vRPN[sz - 2].Cmd = cmVARPOW4;
else
break;

m_vRPN.pop_back();
bOptimized = true;
}
break;

case  cmSUB:
case  cmADD:
if ((m_vRPN[sz - 1].Cmd == cmVAR && m_vRPN[sz - 2].Cmd == cmVAL) ||
(m_vRPN[sz - 1].Cmd == cmVAL && m_vRPN[sz - 2].Cmd == cmVAR) ||
(m_vRPN[sz - 1].Cmd == cmVAL && m_vRPN[sz - 2].Cmd == cmVARMUL) ||
(m_vRPN[sz - 1].Cmd == cmVARMUL && m_vRPN[sz - 2].Cmd == cmVAL) ||
(m_vRPN[sz - 1].Cmd == cmVAR && m_vRPN[sz - 2].Cmd == cmVAR && m_vRPN[sz - 2].Val.ptr == m_vRPN[sz - 1].Val.ptr) ||
(m_vRPN[sz - 1].Cmd == cmVAR && m_vRPN[sz - 2].Cmd == cmVARMUL && m_vRPN[sz - 2].Val.ptr == m_vRPN[sz - 1].Val.ptr) ||
(m_vRPN[sz - 1].Cmd == cmVARMUL && m_vRPN[sz - 2].Cmd == cmVAR && m_vRPN[sz - 2].Val.ptr == m_vRPN[sz - 1].Val.ptr) ||
(m_vRPN[sz - 1].Cmd == cmVARMUL && m_vRPN[sz - 2].Cmd == cmVARMUL && m_vRPN[sz - 2].Val.ptr == m_vRPN[sz - 1].Val.ptr))
{
MUP_ASSERT(
(m_vRPN[sz - 2].Val.ptr == nullptr && m_vRPN[sz - 1].Val.ptr != nullptr) ||
(m_vRPN[sz - 2].Val.ptr != nullptr && m_vRPN[sz - 1].Val.ptr == nullptr) ||
(m_vRPN[sz - 2].Val.ptr == m_vRPN[sz - 1].Val.ptr));

m_vRPN[sz - 2].Cmd = cmVARMUL;
m_vRPN[sz - 2].Val.ptr = (value_type*)((long long)(m_vRPN[sz - 2].Val.ptr) | (long long)(m_vRPN[sz - 1].Val.ptr));    
m_vRPN[sz - 2].Val.data2 += ((a_Oprt == cmSUB) ? -1 : 1) * m_vRPN[sz - 1].Val.data2;  
m_vRPN[sz - 2].Val.data += ((a_Oprt == cmSUB) ? -1 : 1) * m_vRPN[sz - 1].Val.data;   
m_vRPN.pop_back();
bOptimized = true;
}
break;

case  cmMUL:
if ((m_vRPN[sz - 1].Cmd == cmVAR && m_vRPN[sz - 2].Cmd == cmVAL) ||
(m_vRPN[sz - 1].Cmd == cmVAL && m_vRPN[sz - 2].Cmd == cmVAR))
{
m_vRPN[sz - 2].Cmd = cmVARMUL;
m_vRPN[sz - 2].Val.ptr = (value_type*)((long long)(m_vRPN[sz - 2].Val.ptr) | (long long)(m_vRPN[sz - 1].Val.ptr));
m_vRPN[sz - 2].Val.data = m_vRPN[sz - 2].Val.data2 + m_vRPN[sz - 1].Val.data2;
m_vRPN[sz - 2].Val.data2 = 0;
m_vRPN.pop_back();
bOptimized = true;
}
else if (
(m_vRPN[sz - 1].Cmd == cmVAL && m_vRPN[sz - 2].Cmd == cmVARMUL) ||
(m_vRPN[sz - 1].Cmd == cmVARMUL && m_vRPN[sz - 2].Cmd == cmVAL))
{
m_vRPN[sz - 2].Cmd = cmVARMUL;
m_vRPN[sz - 2].Val.ptr = (value_type*)((long long)(m_vRPN[sz - 2].Val.ptr) | (long long)(m_vRPN[sz - 1].Val.ptr));
if (m_vRPN[sz - 1].Cmd == cmVAL)
{
m_vRPN[sz - 2].Val.data *= m_vRPN[sz - 1].Val.data2;
m_vRPN[sz - 2].Val.data2 *= m_vRPN[sz - 1].Val.data2;
}
else
{
m_vRPN[sz - 2].Val.data = m_vRPN[sz - 1].Val.data * m_vRPN[sz - 2].Val.data2;
m_vRPN[sz - 2].Val.data2 = m_vRPN[sz - 1].Val.data2 * m_vRPN[sz - 2].Val.data2;
}
m_vRPN.pop_back();
bOptimized = true;
}
else if (
m_vRPN[sz - 1].Cmd == cmVAR && m_vRPN[sz - 2].Cmd == cmVAR &&
m_vRPN[sz - 1].Val.ptr == m_vRPN[sz - 2].Val.ptr)
{
m_vRPN[sz - 2].Cmd = cmVARPOW2;
m_vRPN.pop_back();
bOptimized = true;
}
break;

case cmDIV:
if (m_vRPN[sz - 1].Cmd == cmVAL && m_vRPN[sz - 2].Cmd == cmVARMUL && m_vRPN[sz - 1].Val.data2 != 0)
{
m_vRPN[sz - 2].Val.data /= m_vRPN[sz - 1].Val.data2;
m_vRPN[sz - 2].Val.data2 /= m_vRPN[sz - 1].Val.data2;
m_vRPN.pop_back();
bOptimized = true;
}
break;

default:
break;
} 
}
}

if (!bOptimized)
{
--m_iStackPos;
SToken tok;
tok.Cmd = a_Oprt;
m_vRPN.push_back(tok);
}
}


void ParserByteCode::AddIfElse(ECmdCode a_Oprt)
{
SToken tok;
tok.Cmd = a_Oprt;
m_vRPN.push_back(tok);
}



void ParserByteCode::AddAssignOp(value_type* a_pVar)
{
--m_iStackPos;

SToken tok;
tok.Cmd = cmASSIGN;
tok.Oprt.ptr = a_pVar;
m_vRPN.push_back(tok);
}



void ParserByteCode::AddFun(generic_callable_type a_pFun, int a_iArgc, bool isFunctionOptimizable)
{
std::size_t sz = m_vRPN.size();
bool optimize = false;

if (isFunctionOptimizable && m_bEnableOptimizer && a_iArgc > 0)
{
if (a_pFun == generic_callable_type{(erased_fun_type)&MathImpl<value_type>::UnaryPlus, nullptr})
return;

optimize = true;

for (int i = 0; i < std::abs(a_iArgc); ++i)
{
if (m_vRPN[sz - i - 1].Cmd != cmVAL)
{
optimize = false;
break;
}
}
}

if (optimize)
{
value_type val = 0;
switch (a_iArgc)
{
case 1:  val = a_pFun.call_fun<1>(m_vRPN[sz - 1].Val.data2); break;
case 2:  val = a_pFun.call_fun<2>(m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 3:  val = a_pFun.call_fun<3>(m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 4:  val = a_pFun.call_fun<4>(m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 5:  val = a_pFun.call_fun<5>(m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 6:  val = a_pFun.call_fun<6>(m_vRPN[sz - 6].Val.data2, m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 7:  val = a_pFun.call_fun<7>(m_vRPN[sz - 7].Val.data2, m_vRPN[sz - 6].Val.data2, m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 8:  val = a_pFun.call_fun<8>(m_vRPN[sz - 8].Val.data2, m_vRPN[sz - 7].Val.data2, m_vRPN[sz - 6].Val.data2, m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 9:  val = a_pFun.call_fun<9>(m_vRPN[sz - 9].Val.data2, m_vRPN[sz - 8].Val.data2, m_vRPN[sz - 7].Val.data2, m_vRPN[sz - 6].Val.data2, m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
case 10: val = a_pFun.call_fun<10>(m_vRPN[sz - 10].Val.data2, m_vRPN[sz - 9].Val.data2, m_vRPN[sz - 8].Val.data2, m_vRPN[sz - 7].Val.data2, m_vRPN[sz - 6].Val.data2, m_vRPN[sz - 5].Val.data2, m_vRPN[sz - 4].Val.data2, m_vRPN[sz - 3].Val.data2, m_vRPN[sz - 2].Val.data2, m_vRPN[sz - 1].Val.data2); break;
default:
throw ParserError(ecINTERNAL_ERROR);
}

m_vRPN.erase(m_vRPN.end() - a_iArgc, m_vRPN.end());

SToken tok;
tok.Cmd = cmVAL;
tok.Val.data = 0;
tok.Val.data2 = val;
tok.Val.ptr = nullptr;
m_vRPN.push_back(tok);
}
else
{
SToken tok;
tok.Cmd = cmFUNC;
tok.Fun.argc = a_iArgc;
tok.Fun.cb = a_pFun;
m_vRPN.push_back(tok);
}

m_iStackPos = m_iStackPos - std::abs(a_iArgc) + 1;
m_iMaxStackSize = std::max(m_iMaxStackSize, (size_t)m_iStackPos);

}



void ParserByteCode::AddBulkFun(generic_callable_type a_pFun, int a_iArgc)
{
m_iStackPos = m_iStackPos - a_iArgc + 1;
m_iMaxStackSize = std::max(m_iMaxStackSize, (size_t)m_iStackPos);

SToken tok;
tok.Cmd = cmFUNC_BULK;
tok.Fun.argc = a_iArgc;
tok.Fun.cb = a_pFun;
m_vRPN.push_back(tok);
}



void ParserByteCode::AddStrFun(generic_callable_type a_pFun, int a_iArgc, int a_iIdx)
{
m_iStackPos = m_iStackPos - a_iArgc + 1;

SToken tok;
tok.Cmd = cmFUNC_STR;
tok.Fun.argc = a_iArgc;
tok.Fun.idx = a_iIdx;
tok.Fun.cb = a_pFun;
m_vRPN.push_back(tok);

m_iMaxStackSize = std::max(m_iMaxStackSize, (size_t)m_iStackPos);
}



void ParserByteCode::Finalize()
{
SToken tok;
tok.Cmd = cmEND;
m_vRPN.push_back(tok);
rpn_type(m_vRPN).swap(m_vRPN);     

std::stack<int> stIf, stElse;
int idx;
for (int i = 0; i < (int)m_vRPN.size(); ++i)
{
switch (m_vRPN[i].Cmd)
{
case cmIF:
stIf.push(i);
break;

case cmELSE:
stElse.push(i);
idx = stIf.top();
stIf.pop();
m_vRPN[idx].Oprt.offset = i - idx;
break;

case cmENDIF:
idx = stElse.top();
stElse.pop();
m_vRPN[idx].Oprt.offset = i - idx;
break;

default:
break;
}
}
}


std::size_t ParserByteCode::GetMaxStackSize() const
{
return m_iMaxStackSize + 1;
}



void ParserByteCode::clear()
{
m_vRPN.clear();
m_iStackPos = 0;
m_iMaxStackSize = 0;
}



void ParserByteCode::AsciiDump()
{
if (!m_vRPN.size())
{
mu::console() << _T("No bytecode available\n");
return;
}

mu::console() << _T("Number of RPN tokens:") << (int)m_vRPN.size() << _T("\n");
for (std::size_t i = 0; i < m_vRPN.size() && m_vRPN[i].Cmd != cmEND; ++i)
{
mu::console() << std::dec << i << _T(" : \t");
switch (m_vRPN[i].Cmd)
{
case cmVAL:   mu::console() << _T("VAL \t");
mu::console() << _T("[") << m_vRPN[i].Val.data2 << _T("]\n");
break;

case cmVAR:   mu::console() << _T("VAR \t");
mu::console() << _T("[ADDR: 0x") << std::hex << m_vRPN[i].Val.ptr << _T("]\n");
break;

case cmVARPOW2: mu::console() << _T("VARPOW2 \t");
mu::console() << _T("[ADDR: 0x") << std::hex << m_vRPN[i].Val.ptr << _T("]\n");
break;

case cmVARPOW3: mu::console() << _T("VARPOW3 \t");
mu::console() << _T("[ADDR: 0x") << std::hex << m_vRPN[i].Val.ptr << _T("]\n");
break;

case cmVARPOW4: mu::console() << _T("VARPOW4 \t");
mu::console() << _T("[ADDR: 0x") << std::hex << m_vRPN[i].Val.ptr << _T("]\n");
break;

case cmVARMUL:  mu::console() << _T("VARMUL \t");
mu::console() << _T("[ADDR: 0x") << std::hex << m_vRPN[i].Val.ptr << _T("]");
mu::console() << _T(" * [") << m_vRPN[i].Val.data << _T("]");
mu::console() << _T(" + [") << m_vRPN[i].Val.data2 << _T("]\n");
break;

case cmFUNC:  mu::console() << _T("CALL\t");
mu::console() << _T("[ARG:") << std::dec << m_vRPN[i].Fun.argc << _T("]");
mu::console() << _T("[ADDR: 0x") << std::hex << reinterpret_cast<void*>(m_vRPN[i].Fun.cb._pRawFun) << _T("]");
mu::console() << _T("[USERDATA: 0x") << std::hex << reinterpret_cast<void*>(m_vRPN[i].Fun.cb._pUserData) << _T("]");
mu::console() << _T("\n");
break;

case cmFUNC_STR:
mu::console() << _T("CALL STRFUNC\t");
mu::console() << _T("[ARG:") << std::dec << m_vRPN[i].Fun.argc << _T("]");
mu::console() << _T("[IDX:") << std::dec << m_vRPN[i].Fun.idx << _T("]");
mu::console() << _T("[ADDR: 0x") << std::hex << reinterpret_cast<void*>(m_vRPN[i].Fun.cb._pRawFun) << _T("]");
mu::console() << _T("[USERDATA: 0x") << std::hex << reinterpret_cast<void*>(m_vRPN[i].Fun.cb._pUserData) << _T("]");
mu::console() << _T("\n");
break;

case cmLT:    mu::console() << _T("LT\n");  break;
case cmGT:    mu::console() << _T("GT\n");  break;
case cmLE:    mu::console() << _T("LE\n");  break;
case cmGE:    mu::console() << _T("GE\n");  break;
case cmEQ:    mu::console() << _T("EQ\n");  break;
case cmNEQ:   mu::console() << _T("NEQ\n"); break;
case cmADD:   mu::console() << _T("ADD\n"); break;
case cmLAND:  mu::console() << _T("&&\n"); break;
case cmLOR:   mu::console() << _T("||\n"); break;
case cmSUB:   mu::console() << _T("SUB\n"); break;
case cmMUL:   mu::console() << _T("MUL\n"); break;
case cmDIV:   mu::console() << _T("DIV\n"); break;
case cmPOW:   mu::console() << _T("POW\n"); break;

case cmIF:    mu::console() << _T("IF\t");
mu::console() << _T("[OFFSET:") << std::dec << m_vRPN[i].Oprt.offset << _T("]\n");
break;

case cmELSE:  mu::console() << _T("ELSE\t");
mu::console() << _T("[OFFSET:") << std::dec << m_vRPN[i].Oprt.offset << _T("]\n");
break;

case cmENDIF: mu::console() << _T("ENDIF\n"); break;

case cmASSIGN:
mu::console() << _T("ASSIGN\t");
mu::console() << _T("[ADDR: 0x") << m_vRPN[i].Oprt.ptr << _T("]\n");
break;

default:      mu::console() << _T("(unknown code: ") << m_vRPN[i].Cmd << _T(")\n");
break;
} 
} 

mu::console() << _T("END") << std::endl;
}
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif