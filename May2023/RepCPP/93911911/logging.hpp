

#ifndef EDGE_CUT_LOGGING_HPP
#define EDGE_CUT_LOGGING_HPP

#ifdef PP_USE_OMP
#define ELPP_THREAD_SAFE
#endif

#pragma GCC system_header
#include <easylogging++.h>

#define EDGE_CUT_LOG_INFO             LOG(INFO)
#define EDGE_CUT_LOG_TRACE            LOG(TRACE)
#define EDGE_CUT_LOG_DEBUG            LOG(DEBUG)
#define EDGE_CUT_LOG_FATAL            LOG(FATAL)
#define EDGE_CUT_LOG_ERROR            LOG(ERROR)
#define EDGE_CUT_LOG_WARNING          LOG(WARNING)
#define EDGE_CUT_LOG_VERBOSE          LOG(VERBOSE)
#define EDGE_CUT_VLOG_IS_ON(str)      VLOG_IS_ON(str)
#define EDGE_CUT_VLOG_ALL(str)        VLOG(str)
#define EDGE_CUT_VLOG(str)            VLOG(str)
#define EDGE_CUT_CHECK(str)           CHECK(str)
#define EDGE_CUT_CHECK_EQ(str1, str2) CHECK_EQ(str1, str2)
#define EDGE_CUT_CHECK_NE(str1, str2) CHECK_NE(str1, str2)
#define EDGE_CUT_CHECK_LT(str1, str2) CHECK_LT(str1, str2)
#define EDGE_CUT_CHECK_GT(str1, str2) CHECK_GT(str1, str2)
#define EDGE_CUT_CHECK_LE(str1, str2) CHECK_LE(str1, str2)
#define EDGE_CUT_CHECK_GE(str1, str2) CHECK_GE(str1, str2)

#endif
