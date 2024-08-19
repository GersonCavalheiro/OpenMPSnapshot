#pragma __vector
_Pragma ("__vector")
#define __vector __new_vector
#define __pixel __new_pixel
#define __bool __new_bool
#define vector new_vector
#define pixel new_pixel
#define bool new_bool
#undef __vector
#define __vector __new_vector
#undef __pixel
#define __pixel __new_pixel
#undef __bool
#define __bool __new_bool
#undef vector
#define vector new_vector
#undef pixel
#define pixel new_pixel
#undef bool
#define bool new_bool
#define __vector	__newer_vector
#define __pixel		__newer_pixel
#define __bool		__newer_bool
#define vector		newer_vector
#define pixel		newer_pixel
#define bool		newer_bool
