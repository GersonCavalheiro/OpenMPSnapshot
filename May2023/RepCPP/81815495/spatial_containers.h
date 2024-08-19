
#pragma once


#if defined(KRATOS_INDEPENDENT)

#define KRATOS_CLASS_POINTER_DEFINITION(variable) \
typedef variable* Pointer

#define KRATOS_WATCH(variable)

#else

#include "includes/define.h"

#endif 

#include "tree.h"
#include "bucket.h"
#include "kd_tree.h"
#include "octree.h"
#include "octree_binary.h"
#include "bins_static.h"
#include "bins_dynamic.h"
#include "bins_dynamic_objects.h"
#include "bins_static_objects.h"