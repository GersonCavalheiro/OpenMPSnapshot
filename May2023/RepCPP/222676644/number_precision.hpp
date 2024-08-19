
#ifndef GLM_GTX_number_precision
#define GLM_GTX_number_precision GLM_VERSION

#include "../glm.hpp"
#include "../gtc/type_precision.hpp"

#if(defined(GLM_MESSAGES) && !defined(glm_ext))
#	pragma message("GLM: GLM_GTX_number_precision extension included")
#endif

namespace glm{
namespace gtx
{


typedef u8			u8vec1;		
typedef u16			u16vec1;    
typedef u32			u32vec1;    
typedef u64			u64vec1;    


typedef f16			f16vec1;    
typedef f32			f32vec1;    
typedef f64			f64vec1;    


typedef f16			f16mat1;    
typedef f16			f16mat1x1;	
typedef f32			f32mat1;	
typedef f32			f32mat1x1;	
typedef f64			f64mat1;	
typedef f64			f64mat1x1;	

}
}

#include "number_precision.inl"

#endif
