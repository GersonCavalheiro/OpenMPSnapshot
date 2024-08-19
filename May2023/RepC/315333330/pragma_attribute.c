#pragma GCC target ("thumb")
#ifndef __thumb__
#error "__thumb__ is not defined"
#endif
#ifdef __thumb2__
#ifndef __ARM_32BIT_STATE
#error  "__ARM_32BIT_STATE is not defined"
#endif
#else 
#ifdef __ARM_32BIT_STATE
#error  "__ARM_32BIT_STATE is defined"
#endif
#endif 
#pragma GCC target ("arm")
#ifdef __thumb__
#error "__thumb__ is defined"
#endif
#if defined (__thumb2__) || defined (__thumb1__)
#error "thumb is defined"
#endif 
#ifndef __ARM_32BIT_STATE
#error  "__ARM_32BIT_STATE is not defined"
#endif
#pragma GCC reset_options
