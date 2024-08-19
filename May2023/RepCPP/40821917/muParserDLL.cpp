

#if defined(MUPARSER_DLL) 

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <windows.h>
#endif

#include <cassert>

#include "muParserDLL.h"
#include "muParser.h"
#include "muParserInt.h"
#include "muParserError.h"

#if _UNICODE
#include <wchar.h>
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif

#define MU_TRY  \
try		\
{

#define MU_CATCH							                       \
}								                           \
catch (muError_t &e)		                               \
{						                                   \
ParserTag *pTag = static_cast<ParserTag*>(a_hParser);  \
pTag->exc = e;                                         \
pTag->bError = true;                                   \
if (pTag->errHandler)                                  \
(pTag->errHandler)(a_hParser);                     \
}					                                       \
catch (...)		                                           \
{		                                                   \
ParserTag *pTag = static_cast<ParserTag*>(a_hParser);  \
pTag->exc = muError_t(mu::ecINTERNAL_ERROR);           \
pTag->bError = true;                                   \
if (pTag->errHandler)                                  \
(pTag->errHandler)(a_hParser);                     \
}



typedef mu::ParserBase::exception_type muError_t;
typedef mu::ParserBase muParser_t;
int g_nBulkSize;


class ParserTag
{
public:
ParserTag(int nType)
: pParser((nType == muBASETYPE_FLOAT)
? (mu::ParserBase*)new mu::Parser()
: (nType == muBASETYPE_INT) ? (mu::ParserBase*)new mu::ParserInt() : nullptr)
, exc()
, errHandler(nullptr)
, bError(false)
, m_nParserType(nType)
{}

~ParserTag()
{
delete pParser;
}

mu::ParserBase* pParser;
mu::ParserBase::exception_type exc;
muErrorHandler_t errHandler;
bool bError;

private:
ParserTag(const ParserTag& ref);
ParserTag& operator=(const ParserTag& ref);

int m_nParserType;
};

static muChar_t s_tmpOutBuf[2048];

template <typename T>
constexpr std::size_t count_of(const T& array) 
{
return (sizeof(array) / sizeof(array[0]));
}



inline muParser_t* AsParser(muParserHandle_t a_hParser)
{
return static_cast<ParserTag*>(a_hParser)->pParser;
}


inline ParserTag* AsParserTag(muParserHandle_t a_hParser)
{
return static_cast<ParserTag*>(a_hParser);
}


#if defined(_WIN32)

BOOL APIENTRY DllMain(HANDLE , DWORD ul_reason_for_call, LPVOID )
{
switch (ul_reason_for_call)
{
case  DLL_PROCESS_ATTACH:
break;

case  DLL_THREAD_ATTACH:
case  DLL_THREAD_DETACH:
case  DLL_PROCESS_DETACH:
break;
}

return TRUE;
}

#endif


API_EXPORT(void) mupSetVarFactory(muParserHandle_t a_hParser, muFacFun_t a_pFactory, void* pUserData)
{
MU_TRY
muParser_t* p(AsParser(a_hParser));
p->SetVarFactory(a_pFactory, pUserData);
MU_CATCH
}



API_EXPORT(muParserHandle_t) mupCreate(int nBaseType)
{
switch (nBaseType)
{
case  muBASETYPE_FLOAT:   return (void*)(new ParserTag(muBASETYPE_FLOAT));
case  muBASETYPE_INT:     return (void*)(new ParserTag(muBASETYPE_INT));
default:                  return nullptr;
}
}



API_EXPORT(void) mupRelease(muParserHandle_t a_hParser)
{
MU_TRY
ParserTag* p = static_cast<ParserTag*>(a_hParser);
delete p;
MU_CATCH
}


API_EXPORT(const muChar_t*) mupGetVersion(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));

#ifndef _UNICODE
snprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), "%s", p->GetVersion().c_str());
#else
swprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), _T("%s"), p->GetVersion().c_str());
#endif

return s_tmpOutBuf;
MU_CATCH
return _T("");
}



API_EXPORT(muFloat_t) mupEval(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
return p->Eval();
MU_CATCH
return 0;
}


API_EXPORT(muFloat_t*) mupEvalMulti(muParserHandle_t a_hParser, int* nNum)
{
MU_TRY
if (nNum == nullptr)
throw std::runtime_error("Argument is null!"); 

muParser_t* const p(AsParser(a_hParser));
return p->Eval(*nNum);
MU_CATCH
return 0;
}


API_EXPORT(void) mupEvalBulk(muParserHandle_t a_hParser, muFloat_t* a_res, int nSize)
{
MU_TRY
muParser_t* p(AsParser(a_hParser));
p->Eval(a_res, nSize);
MU_CATCH
}


API_EXPORT(void) mupSetExpr(muParserHandle_t a_hParser, const muChar_t* a_szExpr)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->SetExpr(a_szExpr);
MU_CATCH
}


API_EXPORT(void) mupRemoveVar(muParserHandle_t a_hParser, const muChar_t* a_szName)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->RemoveVar(a_szName);
MU_CATCH
}



API_EXPORT(void) mupClearVar(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->ClearVar();
MU_CATCH
}



API_EXPORT(void) mupClearConst(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->ClearConst();
MU_CATCH
}



API_EXPORT(void) mupClearOprt(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->ClearOprt();
MU_CATCH
}


API_EXPORT(void) mupClearFun(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->ClearFun();
MU_CATCH
}


API_EXPORT(void) mupDefineFun0(muParserHandle_t a_hParser,
const muChar_t* a_szName,
muFun0_t a_pFun,
muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun1(muParserHandle_t a_hParser,	const muChar_t* a_szName, muFun1_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun2(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun2_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun3(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun3_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun4(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun4_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun5(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun5_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun6(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun6_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun7(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun7_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun8(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun8_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun9(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun9_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFun10(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun10_t a_pFun, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData0(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData0_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData1(muParserHandle_t a_hParser,	const muChar_t* a_szName, muFunUserData1_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData2(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData2_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData3(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData3_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData4(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData4_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData5(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData5_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData6(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData6_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData7(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData7_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData8(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData8_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData9(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData9_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineFunUserData10(muParserHandle_t a_hParser, const muChar_t* a_szName, muFunUserData10_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun0(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun0_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun1(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun1_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun2(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun2_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun3(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun3_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun4(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun4_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun5(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun5_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun6(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun6_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun7(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun7_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun8(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun8_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun9(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun9_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFun10(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFun10_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData0(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData0_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData1(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData1_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData2(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData2_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData3(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData3_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData4(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData4_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData5(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData5_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData6(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData6_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData7(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData7_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData8(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData8_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData9(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData9_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkFunUserData10(muParserHandle_t a_hParser, const muChar_t* a_szName, muBulkFunUserData10_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFun1(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFun1_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFun2(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFun2_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFun3(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFun3_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFun4(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFun4_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFun5(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFun5_t a_pFun)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFunUserData1(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFunUserData1_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFunUserData2(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFunUserData2_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFunUserData3(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFunUserData3_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFunUserData4(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFunUserData4_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineStrFunUserData5(muParserHandle_t a_hParser, const muChar_t* a_szName, muStrFunUserData5_t a_pFun, void* a_pUserData)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, false);
MU_CATCH
}


API_EXPORT(void) mupDefineMultFun(muParserHandle_t a_hParser, const muChar_t* a_szName, muMultFun_t a_pFun,	muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFun(a_szName, a_pFun, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineMultFunUserData(muParserHandle_t a_hParser, const muChar_t* a_szName, muMultFunUserData_t a_pFun, void* a_pUserData, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineFunUserData(a_szName, a_pFun, a_pUserData, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineOprt(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun2_t a_pFun, muInt_t a_nPrec, muInt_t a_nOprtAsct, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineOprt(a_szName, a_pFun,	a_nPrec, (mu::EOprtAssociativity)a_nOprtAsct, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineVar(muParserHandle_t a_hParser, const muChar_t* a_szName,	muFloat_t* a_pVar)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineVar(a_szName, a_pVar);
MU_CATCH
}


API_EXPORT(void) mupDefineBulkVar(muParserHandle_t a_hParser, const muChar_t* a_szName,	muFloat_t* a_pVar)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineVar(a_szName, a_pVar);
MU_CATCH
}


API_EXPORT(void) mupDefineConst(muParserHandle_t a_hParser,	const muChar_t* a_szName, muFloat_t a_fVal)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineConst(a_szName, a_fVal);
MU_CATCH
}


API_EXPORT(void) mupDefineStrConst(muParserHandle_t a_hParser, const muChar_t* a_szName, const muChar_t* a_szVal)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineStrConst(a_szName, a_szVal);
MU_CATCH
}


API_EXPORT(const muChar_t*) mupGetExpr(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));

#ifndef _UNICODE
snprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), "%s", p->GetExpr().c_str());
#else
swprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), _T("%s"), p->GetExpr().c_str());
#endif

return s_tmpOutBuf;
MU_CATCH

return _T("");
}


API_EXPORT(void) mupDefinePostfixOprt(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun1_t a_pOprt, muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefinePostfixOprt(a_szName, a_pOprt, a_bAllowOpt != 0);
MU_CATCH
}


API_EXPORT(void) mupDefineInfixOprt(muParserHandle_t a_hParser, const muChar_t* a_szName, muFun1_t a_pOprt,	muBool_t a_bAllowOpt)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->DefineInfixOprt(a_szName, a_pOprt, a_bAllowOpt != 0);
MU_CATCH
}

API_EXPORT(void) mupDefineNameChars(muParserHandle_t a_hParser, const muChar_t* a_szCharset)
{
muParser_t* const p(AsParser(a_hParser));
p->DefineNameChars(a_szCharset);
}


API_EXPORT(void) mupDefineOprtChars(muParserHandle_t a_hParser,	const muChar_t* a_szCharset)
{
muParser_t* const p(AsParser(a_hParser));
p->DefineOprtChars(a_szCharset);
}


API_EXPORT(void) mupDefineInfixOprtChars(muParserHandle_t a_hParser, const muChar_t* a_szCharset)
{
muParser_t* const p(AsParser(a_hParser));
p->DefineInfixOprtChars(a_szCharset);
}



API_EXPORT(int) mupGetVarNum(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::varmap_type VarMap = p->GetVar();
return (int)VarMap.size();
MU_CATCH

return 0; 
}



API_EXPORT(void) mupGetVar(muParserHandle_t a_hParser, unsigned a_iVar, const muChar_t** a_szName, muFloat_t** a_pVar)
{
static muChar_t  szName[1024];

MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::varmap_type VarMap = p->GetVar();

if (a_iVar >= VarMap.size())
{
*a_szName = 0;
*a_pVar = 0;
return;
}
mu::varmap_type::const_iterator item;

item = VarMap.begin();
for (unsigned i = 0; i < a_iVar; ++i)
++item;

#ifndef _UNICODE
strncpy(szName, item->first.c_str(), count_of(szName));
#else
wcsncpy(szName, item->first.c_str(), count_of(szName));
#endif

szName[count_of(szName) - 1] = 0;

*a_szName = &szName[0];
*a_pVar = item->second;
return;
MU_CATCH

* a_szName = 0;
*a_pVar = 0;
}



API_EXPORT(int) mupGetExprVarNum(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::varmap_type VarMap = p->GetUsedVar();
return (int)VarMap.size();
MU_CATCH

return 0; 
}



API_EXPORT(void) mupGetExprVar(muParserHandle_t a_hParser, unsigned a_iVar, const muChar_t** a_szName, muFloat_t** a_pVar)
{
static muChar_t  szName[1024];

MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::varmap_type VarMap = p->GetUsedVar();

if (a_iVar >= VarMap.size())
{
*a_szName = 0;
*a_pVar = 0;
return;
}
mu::varmap_type::const_iterator item;

item = VarMap.begin();
for (unsigned i = 0; i < a_iVar; ++i)
++item;

#ifndef _UNICODE
strncpy(szName, item->first.c_str(), count_of(szName));
#else
wcsncpy(szName, item->first.c_str(), count_of(szName));
#endif

szName[count_of(szName) - 1] = 0;

*a_szName = &szName[0];
*a_pVar = item->second;
return;
MU_CATCH

* a_szName = 0;
*a_pVar = 0;
}



API_EXPORT(int) mupGetConstNum(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::valmap_type ValMap = p->GetConst();
return (int)ValMap.size();
MU_CATCH

return 0; 
}


API_EXPORT(void) mupSetArgSep(muParserHandle_t a_hParser, const muChar_t cArgSep)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->SetArgSep(cArgSep);
MU_CATCH
}


API_EXPORT(void) mupResetLocale(muParserHandle_t a_hParser)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->ResetLocale();
MU_CATCH
}


API_EXPORT(void) mupSetDecSep(muParserHandle_t a_hParser, const muChar_t cDecSep)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->SetDecSep(cDecSep);
MU_CATCH
}


API_EXPORT(void) mupSetThousandsSep(muParserHandle_t a_hParser, const muChar_t cThousandsSep)
{
MU_TRY
muParser_t* const p(AsParser(a_hParser));
p->SetThousandsSep(cThousandsSep);
MU_CATCH
}


API_EXPORT(void) mupGetConst(muParserHandle_t a_hParser, unsigned a_iVar, const muChar_t** a_pszName, muFloat_t* a_fVal)
{
static muChar_t szName[1024];

MU_TRY
muParser_t* const p(AsParser(a_hParser));
const mu::valmap_type ValMap = p->GetConst();

if (a_iVar >= ValMap.size())
{
*a_pszName = 0;
*a_fVal = 0;
return;
}

mu::valmap_type::const_iterator item;
item = ValMap.begin();
for (unsigned i = 0; i < a_iVar; ++i)
++item;

#ifndef _UNICODE
strncpy(szName, item->first.c_str(), count_of(szName));
#else
wcsncpy(szName, item->first.c_str(), count_of(szName));
#endif

szName[count_of(szName) - 1] = 0;

*a_pszName = &szName[0];
*a_fVal = item->second;
return;

MU_CATCH

* a_pszName = 0;
*a_fVal = 0;
}



API_EXPORT(void) mupAddValIdent(muParserHandle_t a_hParser,	muIdentFun_t a_pFun)
{
MU_TRY
muParser_t* p(AsParser(a_hParser));
p->AddValIdent(a_pFun);
MU_CATCH
}



API_EXPORT(muBool_t) mupError(muParserHandle_t a_hParser)
{
bool bError(AsParserTag(a_hParser)->bError);
AsParserTag(a_hParser)->bError = false;
return bError;
}



API_EXPORT(void) mupErrorReset(muParserHandle_t a_hParser)
{
AsParserTag(a_hParser)->bError = false;
}


API_EXPORT(void) mupSetErrorHandler(muParserHandle_t a_hParser, muErrorHandler_t a_pHandler)
{
AsParserTag(a_hParser)->errHandler = a_pHandler;
}



API_EXPORT(const muChar_t*) mupGetErrorMsg(muParserHandle_t a_hParser)
{
ParserTag* const p(AsParserTag(a_hParser));
const muChar_t* pMsg = p->exc.GetMsg().c_str();

#ifndef _UNICODE
snprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), "%s", pMsg);
#else
swprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), _T("%s"), pMsg);
#endif

return s_tmpOutBuf;
}



API_EXPORT(const muChar_t*) mupGetErrorToken(muParserHandle_t a_hParser)
{
ParserTag* const p(AsParserTag(a_hParser));
const muChar_t* pToken = p->exc.GetToken().c_str();

#ifndef _UNICODE
snprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), "%s", pToken);
#else
swprintf(s_tmpOutBuf, count_of(s_tmpOutBuf), _T("%s"), pToken);
#endif

return s_tmpOutBuf;
}



API_EXPORT(int) mupGetErrorCode(muParserHandle_t a_hParser)
{
return AsParserTag(a_hParser)->exc.GetCode();
}



API_EXPORT(int) mupGetErrorPos(muParserHandle_t a_hParser)
{
return (int)AsParserTag(a_hParser)->exc.GetPos();
}


API_EXPORT(muFloat_t*) mupCreateVar()
{
return new muFloat_t(0);
}


API_EXPORT(void) mupReleaseVar(muFloat_t* ptr)
{
delete ptr;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif      
