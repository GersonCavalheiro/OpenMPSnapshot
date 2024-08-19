

#include "muParserCallback.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 26812) 
#endif




namespace mu
{
static constexpr int CALLBACK_INTERNAL_VAR_ARGS         = 1 << 14;
static constexpr int CALLBACK_INTERNAL_FIXED_ARGS_MASK  = 0xf;
static constexpr int CALLBACK_INTERNAL_WITH_USER_DATA	= 1 << 13;

struct CbWithUserData
{
void*	pFun;
void* 	pUserData;
};


ParserCallback::ParserCallback(fun_type0 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(0)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type1 a_pFun, bool a_bAllowOpti, int a_iPrec, ECmdCode a_iCode)
:m_pFun((void*)a_pFun)
, m_iArgc(1)
, m_iPri(a_iPrec)
, m_eOprtAsct(oaNONE)
, m_iCode(a_iCode)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type1 a_pFun, bool a_bAllowOpti)
: ParserCallback(a_pFun, a_bAllowOpti, -1, cmFUNC)
{}



ParserCallback::ParserCallback(fun_type2 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(2)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}



ParserCallback::ParserCallback(fun_type2 a_pFun,
bool a_bAllowOpti,
int a_iPrec,
EOprtAssociativity a_eOprtAsct)
:m_pFun((void*)a_pFun)
, m_iArgc(2)
, m_iPri(a_iPrec)
, m_eOprtAsct(a_eOprtAsct)
, m_iCode(cmOPRT_BIN)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type3 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(3)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type4 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(4)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type5 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(5)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type6 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(6)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type7 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(7)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type8 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(8)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type9 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(9)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_type10 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(10)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type0 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(0 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type1 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(1 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type2 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(2 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type3 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(3 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type4 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(4 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type5 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(5 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type6 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(6 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type7 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(7 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type8 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(8 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type9 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(9 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(fun_userdata_type10 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(10 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type0 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(0)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type1 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(1)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}



ParserCallback::ParserCallback(bulkfun_type2 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(2)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type3 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(3)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type4 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(4)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type5 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(5)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type6 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(6)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type7 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(7)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type8 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(8)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type9 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(9)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_type10 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(10)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type0 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(0 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type1 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(1 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type2 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(2 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type3 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(3 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type4 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(4 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type5 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(5 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type6 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(6 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type7 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(7 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type8 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(8 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type9 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(9 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(bulkfun_userdata_type10 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(10 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_BULK)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(multfun_type a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(CALLBACK_INTERNAL_VAR_ARGS)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(multfun_userdata_type a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(CALLBACK_INTERNAL_VAR_ARGS | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC)
, m_iType(tpDBL)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_type1 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(0)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_type2 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(1)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_type3 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(2)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_type4 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(3)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_type5 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(4)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}

ParserCallback::ParserCallback(strfun_type6 a_pFun, bool a_bAllowOpti)
:m_pFun((void*)a_pFun)
, m_iArgc(5)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type1 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(0 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type2 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(1 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type3 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(2 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type4 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(3 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type5 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{reinterpret_cast<void*>(a_pFun), a_pUserData})
, m_iArgc(4 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback(strfun_userdata_type6 a_pFun, void* a_pUserData, bool a_bAllowOpti)
:m_pFun(new CbWithUserData{ reinterpret_cast<void*>(a_pFun), a_pUserData })
, m_iArgc(5 | CALLBACK_INTERNAL_WITH_USER_DATA)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmFUNC_STR)
, m_iType(tpSTR)
, m_bAllowOpti(a_bAllowOpti)
{}


ParserCallback::ParserCallback()
:m_pFun(0)
, m_iArgc(0)
, m_iPri(-1)
, m_eOprtAsct(oaNONE)
, m_iCode(cmUNKNOWN)
, m_iType(tpVOID)
, m_bAllowOpti(0)
{}



ParserCallback::ParserCallback(const ParserCallback& ref)
:ParserCallback()
{
Assign(ref);
}

ParserCallback & ParserCallback::operator=(const ParserCallback & ref)
{
Assign(ref);
return *this;
}


ParserCallback::~ParserCallback()
{
if (m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA)
delete reinterpret_cast<CbWithUserData*>(m_pFun);
}



void ParserCallback::Assign(const ParserCallback& ref)
{
if (this == &ref)
return;

if (m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA) {
delete reinterpret_cast<CbWithUserData*>(m_pFun);
m_pFun = nullptr;
}

if (ref.m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA)
m_pFun = new CbWithUserData(*reinterpret_cast<CbWithUserData*>(ref.m_pFun));
else
m_pFun = ref.m_pFun;
m_iArgc = ref.m_iArgc;
m_bAllowOpti = ref.m_bAllowOpti;
m_iCode = ref.m_iCode;
m_iType = ref.m_iType;
m_iPri = ref.m_iPri;
m_eOprtAsct = ref.m_eOprtAsct;
}



ParserCallback* ParserCallback::Clone() const
{
return new ParserCallback(*this);
}



bool ParserCallback::IsOptimizable() const
{
return m_bAllowOpti;
}



void* ParserCallback::GetAddr() const
{
if (m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA)
return reinterpret_cast<CbWithUserData*>(m_pFun)->pFun;
else
return m_pFun;
}



void* ParserCallback::GetUserData() const
{
if (m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA)
return reinterpret_cast<CbWithUserData*>(m_pFun)->pUserData;
else
return nullptr;
}



bool ParserCallback::IsValid() const
{
return GetAddr() != nullptr
&& !((m_iArgc & CALLBACK_INTERNAL_WITH_USER_DATA)
&& GetUserData() == nullptr);
}



ECmdCode  ParserCallback::GetCode() const
{
return m_iCode;
}


ETypeCode ParserCallback::GetType() const
{
return m_iType;
}



int ParserCallback::GetPri()  const
{
return m_iPri;
}



EOprtAssociativity ParserCallback::GetAssociativity() const
{
return m_eOprtAsct;
}



int ParserCallback::GetArgc() const
{
return (m_iArgc & CALLBACK_INTERNAL_VAR_ARGS) ? -1 : (m_iArgc & CALLBACK_INTERNAL_FIXED_ARGS_MASK);
}
} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
