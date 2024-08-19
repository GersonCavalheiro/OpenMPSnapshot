

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_aligned_space_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_aligned_space_H
#pragma message("TBB Warning: tbb/aligned_space.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_aligned_space_H
#define __TBB_aligned_space_H

#define __TBB_aligned_space_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include "tbb_machine.h"

namespace tbb {


template<typename T,size_t N=1>
class __TBB_DEPRECATED_VERBOSE_MSG("tbb::aligned_space is deprecated, use std::aligned_storage") aligned_space {
private:
typedef __TBB_TypeWithAlignmentAtLeastAsStrict(T) element_type;
element_type array[(sizeof(T)*N+sizeof(element_type)-1)/sizeof(element_type)];
public:
T* begin() const {return internal::punned_cast<T*>(this);}

T* end() const {return begin()+N;}
};

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_aligned_space_H_include_area

#endif 
