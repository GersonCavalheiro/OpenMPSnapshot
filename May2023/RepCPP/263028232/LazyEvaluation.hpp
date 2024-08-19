#pragma once

#include "CustomLazyOP.hpp"
#include "LazyTranspose.hpp"
#include "LazyMatrixMultiplication.hpp"

#include "binary/Add.hpp"
#include "binary/Sub.hpp"
#include "binary/Mul.hpp"
#include "binary/Div.hpp"
#include "unary/Neg.hpp"

#include "binary/Equal.hpp"
#include "binary/NotEqual.hpp"
#include "binary/GreaterThan.hpp"
#include "binary/LessThan.hpp"
#include "binary/GreaterEqualThan.hpp"
#include "binary/LessEqualThan.hpp"

#include "binary/Maximum.hpp"
#include "binary/Minimum.hpp"
#include "binary/Pow.hpp"

#include "binary/Reshape.hpp"

#include "unary/Log.hpp"
#include "unary/Sqrt.hpp"
#include "unary/Square.hpp"
#include "unary/Exp.hpp"
#include "unary/Abs.hpp"

#include "unary/Sin.hpp"
#include "unary/Cos.hpp"
#include "unary/Tan.hpp"
#include "unary/Cot.hpp"
#include "unary/Sinh.hpp"
#include "unary/Cosh.hpp"
#include "unary/Tanh.hpp"

#include "reduction/Sum.hpp"
#include "reduction/Mean.hpp"
#include "reduction/Min.hpp"
#include "reduction/Max.hpp"
#include "reduction/Argmin.hpp"
#include "reduction/Argmax.hpp"