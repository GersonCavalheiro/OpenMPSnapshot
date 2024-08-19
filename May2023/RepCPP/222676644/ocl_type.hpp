
#ifndef GLM_GTX_ocl_type
#define GLM_GTX_ocl_type GLM_VERSION

#include "../glm.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_ocl_type extension included")
#endif

namespace glm{
namespace gtx
{


typedef detail::int8						cl_char;		
typedef detail::int16						cl_short;		
typedef detail::int32						cl_int;			
typedef detail::int64						cl_long;		

typedef detail::uint8						cl_uchar;		
typedef detail::uint16						cl_ushort;		
typedef detail::uint32						cl_uint;		
typedef detail::uint64						cl_ulong;		

typedef detail::float16						cl_half;	
typedef detail::float32						cl_float;	


typedef detail::int8						cl_char1;		
typedef detail::int16						cl_short1;		
typedef detail::int32						cl_int1;			
typedef detail::int64						cl_long1;		

typedef detail::uint8						cl_uchar1;		
typedef detail::uint16						cl_ushort1;		
typedef detail::uint32						cl_uint1;		
typedef detail::uint64						cl_ulong1;		

typedef detail::float32						cl_float1;	


typedef detail::tvec2<detail::int8>			cl_char2;		
typedef detail::tvec2<detail::int16>		cl_short2;		
typedef detail::tvec2<detail::int32>		cl_int2;			
typedef detail::tvec2<detail::int64>		cl_long2;		

typedef detail::tvec2<detail::uint8>		cl_uchar2;		
typedef detail::tvec2<detail::uint16>		cl_ushort2;		
typedef detail::tvec2<detail::uint32>		cl_uint2;		
typedef detail::tvec2<detail::uint64>		cl_ulong2;		

typedef detail::tvec2<detail::float32>		cl_float2;	


typedef detail::tvec3<detail::int8>			cl_char3;		
typedef detail::tvec3<detail::int16>		cl_short3;		
typedef detail::tvec3<detail::int32>		cl_int3;			
typedef detail::tvec3<detail::int64>		cl_long3;		

typedef detail::tvec3<detail::uint8>		cl_uchar3;		
typedef detail::tvec3<detail::uint16>		cl_ushort3;		
typedef detail::tvec3<detail::uint32>		cl_uint3;		
typedef detail::tvec3<detail::uint64>		cl_ulong3;		

typedef detail::tvec3<detail::float32>		cl_float3;	


typedef detail::tvec4<detail::int8>			cl_char4;		
typedef detail::tvec4<detail::int16>		cl_short4;		
typedef detail::tvec4<detail::int32>		cl_int4;			
typedef detail::tvec4<detail::int64>		cl_long4;		
typedef detail::tvec4<detail::uint8>		cl_uchar4;		
typedef detail::tvec4<detail::uint16>		cl_ushort4;		
typedef detail::tvec4<detail::uint32>		cl_uint4;		
typedef detail::tvec4<detail::uint64>		cl_ulong4;		

typedef detail::tvec4<detail::float32>		cl_float4;	

}
}

#include "ocl_type.inl"

#endif
