
#ifndef CPPTL_JSON_FEATURES_H_INCLUDED
#define CPPTL_JSON_FEATURES_H_INCLUDED

#if !defined(JSON_IS_AMALGAMATION)
#include "forwards.h"
#endif 

#pragma pack(push, 8)

namespace Json {


class JSON_API Features {
public:

static Features all();


static Features strictMode();


Features();

bool allowComments_;

bool strictRoot_;

bool allowDroppedNullPlaceholders_;

bool allowNumericKeys_;
};

} 

#pragma pack(pop)

#endif 
