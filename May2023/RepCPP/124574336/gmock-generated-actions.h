




#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_ACTIONS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_ACTIONS_H_

#include <memory>
#include <utility>

#include "gmock/gmock-actions.h"
#include "gmock/internal/gmock-port.h"

#include "gmock/internal/custom/gmock-generated-actions.h"


#define GMOCK_INTERNAL_DECL_HAS_1_TEMPLATE_PARAMS(kind0, name0) kind0 name0
#define GMOCK_INTERNAL_DECL_HAS_2_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1) kind0 name0, kind1 name1
#define GMOCK_INTERNAL_DECL_HAS_3_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2) kind0 name0, kind1 name1, kind2 name2
#define GMOCK_INTERNAL_DECL_HAS_4_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3) kind0 name0, kind1 name1, kind2 name2, \
kind3 name3
#define GMOCK_INTERNAL_DECL_HAS_5_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4) kind0 name0, kind1 name1, \
kind2 name2, kind3 name3, kind4 name4
#define GMOCK_INTERNAL_DECL_HAS_6_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5) kind0 name0, \
kind1 name1, kind2 name2, kind3 name3, kind4 name4, kind5 name5
#define GMOCK_INTERNAL_DECL_HAS_7_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6) kind0 name0, kind1 name1, kind2 name2, kind3 name3, kind4 name4, \
kind5 name5, kind6 name6
#define GMOCK_INTERNAL_DECL_HAS_8_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7) kind0 name0, kind1 name1, kind2 name2, kind3 name3, \
kind4 name4, kind5 name5, kind6 name6, kind7 name7
#define GMOCK_INTERNAL_DECL_HAS_9_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7, kind8, name8) kind0 name0, kind1 name1, kind2 name2, \
kind3 name3, kind4 name4, kind5 name5, kind6 name6, kind7 name7, \
kind8 name8
#define GMOCK_INTERNAL_DECL_HAS_10_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1, kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6, kind7, name7, kind8, name8, kind9, name9) kind0 name0, \
kind1 name1, kind2 name2, kind3 name3, kind4 name4, kind5 name5, \
kind6 name6, kind7 name7, kind8 name8, kind9 name9

#define GMOCK_INTERNAL_LIST_HAS_1_TEMPLATE_PARAMS(kind0, name0) name0
#define GMOCK_INTERNAL_LIST_HAS_2_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1) name0, name1
#define GMOCK_INTERNAL_LIST_HAS_3_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2) name0, name1, name2
#define GMOCK_INTERNAL_LIST_HAS_4_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3) name0, name1, name2, name3
#define GMOCK_INTERNAL_LIST_HAS_5_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4) name0, name1, name2, name3, \
name4
#define GMOCK_INTERNAL_LIST_HAS_6_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5) name0, name1, \
name2, name3, name4, name5
#define GMOCK_INTERNAL_LIST_HAS_7_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6) name0, name1, name2, name3, name4, name5, name6
#define GMOCK_INTERNAL_LIST_HAS_8_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7) name0, name1, name2, name3, name4, name5, name6, name7
#define GMOCK_INTERNAL_LIST_HAS_9_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7, kind8, name8) name0, name1, name2, name3, name4, name5, \
name6, name7, name8
#define GMOCK_INTERNAL_LIST_HAS_10_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1, kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6, kind7, name7, kind8, name8, kind9, name9) name0, name1, name2, \
name3, name4, name5, name6, name7, name8, name9

#define GMOCK_INTERNAL_DECL_TYPE_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DECL_TYPE_AND_1_VALUE_PARAMS(p0) , typename p0##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_2_VALUE_PARAMS(p0, p1) , \
typename p0##_type, typename p1##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_3_VALUE_PARAMS(p0, p1, p2) , \
typename p0##_type, typename p1##_type, typename p2##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_4_VALUE_PARAMS(p0, p1, p2, p3) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8, p9) , typename p0##_type, typename p1##_type, \
typename p2##_type, typename p3##_type, typename p4##_type, \
typename p5##_type, typename p6##_type, typename p7##_type, \
typename p8##_type, typename p9##_type

#define GMOCK_INTERNAL_INIT_AND_0_VALUE_PARAMS()\
()
#define GMOCK_INTERNAL_INIT_AND_1_VALUE_PARAMS(p0)\
(p0##_type gmock_p0) : p0(::std::move(gmock_p0))
#define GMOCK_INTERNAL_INIT_AND_2_VALUE_PARAMS(p0, p1)\
(p0##_type gmock_p0, p1##_type gmock_p1) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1))
#define GMOCK_INTERNAL_INIT_AND_3_VALUE_PARAMS(p0, p1, p2)\
(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2))
#define GMOCK_INTERNAL_INIT_AND_4_VALUE_PARAMS(p0, p1, p2, p3)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3))
#define GMOCK_INTERNAL_INIT_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4))
#define GMOCK_INTERNAL_INIT_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4)), \
p5(::std::move(gmock_p5))
#define GMOCK_INTERNAL_INIT_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4)), \
p5(::std::move(gmock_p5)), p6(::std::move(gmock_p6))
#define GMOCK_INTERNAL_INIT_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4)), \
p5(::std::move(gmock_p5)), p6(::std::move(gmock_p6)), \
p7(::std::move(gmock_p7))
#define GMOCK_INTERNAL_INIT_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, \
p8##_type gmock_p8) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4)), \
p5(::std::move(gmock_p5)), p6(::std::move(gmock_p6)), \
p7(::std::move(gmock_p7)), p8(::std::move(gmock_p8))
#define GMOCK_INTERNAL_INIT_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, p8##_type gmock_p8, \
p9##_type gmock_p9) : p0(::std::move(gmock_p0)), \
p1(::std::move(gmock_p1)), p2(::std::move(gmock_p2)), \
p3(::std::move(gmock_p3)), p4(::std::move(gmock_p4)), \
p5(::std::move(gmock_p5)), p6(::std::move(gmock_p6)), \
p7(::std::move(gmock_p7)), p8(::std::move(gmock_p8)), \
p9(::std::move(gmock_p9))

#define GMOCK_INTERNAL_DEFN_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DEFN_AND_1_VALUE_PARAMS(p0) p0##_type p0;
#define GMOCK_INTERNAL_DEFN_AND_2_VALUE_PARAMS(p0, p1) p0##_type p0; \
p1##_type p1;
#define GMOCK_INTERNAL_DEFN_AND_3_VALUE_PARAMS(p0, p1, p2) p0##_type p0; \
p1##_type p1; p2##_type p2;
#define GMOCK_INTERNAL_DEFN_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0##_type p0; \
p1##_type p1; p2##_type p2; p3##_type p3;
#define GMOCK_INTERNAL_DEFN_AND_5_VALUE_PARAMS(p0, p1, p2, p3, \
p4) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4;
#define GMOCK_INTERNAL_DEFN_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, \
p5) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5;
#define GMOCK_INTERNAL_DEFN_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5; p6##_type p6;
#define GMOCK_INTERNAL_DEFN_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5; p6##_type p6; p7##_type p7;
#define GMOCK_INTERNAL_DEFN_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; \
p4##_type p4; p5##_type p5; p6##_type p6; p7##_type p7; p8##_type p8;
#define GMOCK_INTERNAL_DEFN_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; \
p4##_type p4; p5##_type p5; p6##_type p6; p7##_type p7; p8##_type p8; \
p9##_type p9;

#define GMOCK_INTERNAL_LIST_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_LIST_AND_1_VALUE_PARAMS(p0) p0
#define GMOCK_INTERNAL_LIST_AND_2_VALUE_PARAMS(p0, p1) p0, p1
#define GMOCK_INTERNAL_LIST_AND_3_VALUE_PARAMS(p0, p1, p2) p0, p1, p2
#define GMOCK_INTERNAL_LIST_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0, p1, p2, p3
#define GMOCK_INTERNAL_LIST_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) p0, p1, \
p2, p3, p4
#define GMOCK_INTERNAL_LIST_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) p0, \
p1, p2, p3, p4, p5
#define GMOCK_INTERNAL_LIST_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0, p1, p2, p3, p4, p5, p6
#define GMOCK_INTERNAL_LIST_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0, p1, p2, p3, p4, p5, p6, p7
#define GMOCK_INTERNAL_LIST_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0, p1, p2, p3, p4, p5, p6, p7, p8
#define GMOCK_INTERNAL_LIST_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0, p1, p2, p3, p4, p5, p6, p7, p8, p9

#define GMOCK_INTERNAL_LIST_TYPE_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_LIST_TYPE_AND_1_VALUE_PARAMS(p0) , p0##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_2_VALUE_PARAMS(p0, p1) , p0##_type, \
p1##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_3_VALUE_PARAMS(p0, p1, p2) , p0##_type, \
p1##_type, p2##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_4_VALUE_PARAMS(p0, p1, p2, p3) , \
p0##_type, p1##_type, p2##_type, p3##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) , \
p0##_type, p1##_type, p2##_type, p3##_type, p4##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) , \
p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, p5##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, p5##_type, \
p6##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, p8##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8, p9) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, p8##_type, p9##_type

#define GMOCK_INTERNAL_DECL_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DECL_AND_1_VALUE_PARAMS(p0) p0##_type p0
#define GMOCK_INTERNAL_DECL_AND_2_VALUE_PARAMS(p0, p1) p0##_type p0, \
p1##_type p1
#define GMOCK_INTERNAL_DECL_AND_3_VALUE_PARAMS(p0, p1, p2) p0##_type p0, \
p1##_type p1, p2##_type p2
#define GMOCK_INTERNAL_DECL_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0##_type p0, \
p1##_type p1, p2##_type p2, p3##_type p3
#define GMOCK_INTERNAL_DECL_AND_5_VALUE_PARAMS(p0, p1, p2, p3, \
p4) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4
#define GMOCK_INTERNAL_DECL_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, \
p5) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5
#define GMOCK_INTERNAL_DECL_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5, p6##_type p6
#define GMOCK_INTERNAL_DECL_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5, p6##_type p6, p7##_type p7
#define GMOCK_INTERNAL_DECL_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, p8##_type p8
#define GMOCK_INTERNAL_DECL_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, p8##_type p8, \
p9##_type p9

#define GMOCK_INTERNAL_COUNT_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_COUNT_AND_1_VALUE_PARAMS(p0) P
#define GMOCK_INTERNAL_COUNT_AND_2_VALUE_PARAMS(p0, p1) P2
#define GMOCK_INTERNAL_COUNT_AND_3_VALUE_PARAMS(p0, p1, p2) P3
#define GMOCK_INTERNAL_COUNT_AND_4_VALUE_PARAMS(p0, p1, p2, p3) P4
#define GMOCK_INTERNAL_COUNT_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) P5
#define GMOCK_INTERNAL_COUNT_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) P6
#define GMOCK_INTERNAL_COUNT_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6) P7
#define GMOCK_INTERNAL_COUNT_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) P8
#define GMOCK_INTERNAL_COUNT_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) P9
#define GMOCK_INTERNAL_COUNT_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) P10

#define GMOCK_ACTION_CLASS_(name, value_params)\
GTEST_CONCAT_TOKEN_(name##Action, GMOCK_INTERNAL_COUNT_##value_params)

#define ACTION_TEMPLATE(name, template_params, value_params)\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
class GMOCK_ACTION_CLASS_(name, value_params) {\
public:\
explicit GMOCK_ACTION_CLASS_(name, value_params)\
GMOCK_INTERNAL_INIT_##value_params {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
explicit gmock_Impl GMOCK_INTERNAL_INIT_##value_params {}\
return_type Perform(const args_type& args) override {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>\
return_type gmock_PerformImpl(GMOCK_ACTION_ARG_TYPES_AND_NAMES_) const;\
GMOCK_INTERNAL_DEFN_##value_params\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(\
new gmock_Impl<F>(GMOCK_INTERNAL_LIST_##value_params));\
}\
GMOCK_INTERNAL_DEFN_##value_params\
};\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
inline GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params> name(\
GMOCK_INTERNAL_DECL_##value_params) {\
return GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params>(\
GMOCK_INTERNAL_LIST_##value_params);\
}\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
template <typename F>\
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>\
typename ::testing::internal::Function<F>::Result\
GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params>::gmock_Impl<F>::\
gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const


namespace testing {


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif


ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
return internal::InvokeArgument(::std::get<k>(args));
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(p0)) {
return internal::InvokeArgument(::std::get<k>(args), p0);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_2_VALUE_PARAMS(p0, p1)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_3_VALUE_PARAMS(p0, p1, p2)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_4_VALUE_PARAMS(p0, p1, p2, p3)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4, p5);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4, p5,
p6);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4, p5,
p6, p7);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4, p5,
p6, p7, p8);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)) {
return internal::InvokeArgument(::std::get<k>(args), p0, p1, p2, p3, p4, p5,
p6, p7, p8, p9);
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

}  

#endif  
