
#ifndef GLM_EXT_INCLUDED
#define GLM_EXT_INCLUDED

#if(defined(GLM_MESSAGES) && !defined(GLM_MESSAGE_EXT_INCLUDED_DISPLAYED))
#	define GLM_MESSAGE_EXT_INCLUDED_DISPLAYED
#	pragma message("GLM: All extensions included (not recommanded)")
#endif

#include "./gtc/constants.hpp"
#include "./gtc/epsilon.hpp"
#include "./gtc/matrix_access.hpp"
#include "./gtc/matrix_integer.hpp"
#include "./gtc/matrix_inverse.hpp"
#include "./gtc/matrix_transform.hpp"
#include "./gtc/noise.hpp"
#include "./gtc/packing.hpp"
#include "./gtc/quaternion.hpp"
#include "./gtc/random.hpp"
#include "./gtc/reciprocal.hpp"
#include "./gtc/type_precision.hpp"
#include "./gtc/type_ptr.hpp"
#include "./gtc/ulp.hpp"

#include "./gtx/associated_min_max.hpp"
#include "./gtx/bit.hpp"
#include "./gtx/closest_point.hpp"
#include "./gtx/color_space.hpp"
#include "./gtx/color_space_YCoCg.hpp"
#include "./gtx/compatibility.hpp"
#include "./gtx/component_wise.hpp"
#include "./gtx/dual_quaternion.hpp"
#include "./gtx/euler_angles.hpp"
#include "./gtx/extend.hpp"
#include "./gtx/extented_min_max.hpp"
#include "./gtx/fast_exponential.hpp"
#include "./gtx/fast_square_root.hpp"
#include "./gtx/fast_trigonometry.hpp"
#include "./gtx/gradient_paint.hpp"
#include "./gtx/handed_coordinate_space.hpp"
#include "./gtx/inertia.hpp"
#include "./gtx/int_10_10_10_2.hpp"
#include "./gtx/integer.hpp"
#include "./gtx/intersect.hpp"
#include "./gtx/log_base.hpp"
#include "./gtx/matrix_cross_product.hpp"
#include "./gtx/matrix_interpolation.hpp"
#include "./gtx/matrix_major_storage.hpp"
#include "./gtx/matrix_operation.hpp"
#include "./gtx/matrix_query.hpp"
#include "./gtx/mixed_product.hpp"
#include "./gtx/multiple.hpp"
#include "./gtx/norm.hpp"
#include "./gtx/normal.hpp"
#include "./gtx/normalize_dot.hpp"
#include "./gtx/number_precision.hpp"
#include "./gtx/optimum_pow.hpp"
#include "./gtx/orthonormalize.hpp"
#include "./gtx/perpendicular.hpp"
#include "./gtx/polar_coordinates.hpp"
#include "./gtx/projection.hpp"
#include "./gtx/quaternion.hpp"
#include "./gtx/raw_data.hpp"
#include "./gtx/rotate_vector.hpp"
#include "./gtx/spline.hpp"
#include "./gtx/std_based_type.hpp"
#if(!(GLM_COMPILER & GLM_COMPILER_CUDA))
#	include "./gtx/string_cast.hpp"
#endif
#include "./gtx/transform.hpp"
#include "./gtx/transform2.hpp"
#include "./gtx/vec1.hpp"
#include "./gtx/vector_angle.hpp"
#include "./gtx/vector_query.hpp"
#include "./gtx/wrap.hpp"

#if(GLM_ARCH & GLM_ARCH_SSE2)
#	include "./gtx/simd_vec4.hpp"
#	include "./gtx/simd_mat4.hpp"
#endif

#endif 