/*
The MIT License (MIT)

Copyright (c) 2020 Alec Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

// @Copyright Alec Miller 2030
// Please include this 
#include <TargetConditionals.h>

#if TARGET_OS_WIN32 || TARGET_OS_UNIX
    #include <intrin.h>
    #define qSSE 1

    //#warning "compiling for Win"

#elif TARGET_OS_OSX || TARGET_CPU_X86_64
    #include <Accelerate/Accelerate.h>
    #include <x86intrin.h>
    #define qSSE 1

    //#warning "compiling for macOS or Sim"

#elif TARGET_OS_IOS
    #define qSSE 0
    #include <Accelerate/Accelerate.h>
    #include <arm_neon.h>

    //#warning "compiling for iOS"

#else
    #error "unknown platform"
#endif

//----------------------------------------------------------------
// define finline, so it can be undefed when embedded in a debug builds
#if TARGET_OS_WIN32
    #define finline __forceinline
#else
    #define finline inline __attribute__((__always_inline__, __nodebug__))
#endif

// TDOO: undef here in debug, to force these into an optimized .cpp

//----------------------------------------------------------------
#if qSSE

// typedef sse __m1128 to Neon type since it's clearr that they store, no typesafety though
typedef __m128  float32x4_t;
// typedef __m128i int8x16_t;
// typedef __m128i int16x8_t;
// typedef __m128i int32x4_t;

#define _mm_fixzero_ps(a,b)  _mm_and_ps(a, _mm_cmpneq_ps(b, _mm_setzero_ps()))

// already high-precision calls for sqrt
#define _mm_sqrthp_ss(a)     _mm_sqrt_ss(a)                     
#define _mm_sqrthp_ps(a)     _mm_sqrt_ps(a)

finline float32x4_t _mm_rsqrthp_ps(const float32x4_t& a) 
{
    static const float32x4_t kHalf  = _mm_set1_ps(0.5f);
    static const float32x4_t kThree = _mm_set1_ps(3.0f);
    
    //for the reciprocal square root, it looks like this: (doesn't handle 0 -> Inf -> Nan), mix in min/max)
    // this ix 2x precision
	//x = rsqrt_approx(c);
    //x *= 0.5*(3 - x*x*c); // refinement
	
    float32x4_t low;
    low = _mm_rsqrt_ps(a); // low precision
    
    // zero out any elements that started out zero (0 -> 0)
    low = _mm_fixzero_ps(low, a); 
    
    // this is already rolled into Neon wrapper
    low = _mm_mul_ps(low, _mm_mul_ps(kHalf, 
            _mm_sub_ps(kThree, _mm_mul_ps(a, _mm_mul_ps(low,low)) )));
                     
	return low;
}

finline float32x4_t _mm_rcphp_ps(const float32x4_t& a) 
{
    static const float32x4_t kTwo = _mm_set1_ps(2.0f);
    
    // http://www.virtualdub.org/blog/pivot/entry.php?id=229#body (doesn't handle 0 -> Inf, min in min/max)
	// 20-bit precision
	//x = reciprocal_approx(c);
    //x' = x * (2 - x * c);
    
	float32x4_t low = _mm_rcp_ps(a);
    
    // zero out any elements that started out 0 (0 -> 0)
    low = _mm_fixzero_ps(low, a); 
    
    // this is already rolled into Neon wrapper
    low = _mm_mul_ps(low, _mm_sub_ps(kTwo, _mm_mul_ps(low,a) ));

	return low;
}

#define _mm_rsqrthp_ss(a) _mm_setx_ps(a, _mm_rsqrthp_ps(a))                   
#define _mm_rcphp_ss(a)   _mm_setx_ps(a, _mm_rcphp_ps(a))

#else
//----------------------------------------------------------------

#define _mm_set_ps(d,c,b,a)  (float32x4_t){a,b,c,d}
#define _mm_setr_ps(d,c,b,a) (float32x4_t){d,c,b,a}
#define _mm_set1_ps(x)      vdupq_n_f32(x)

// replicate a lane into a new vector
#define _mm_setxxxx_ps(v) vdup_lane_f32(v, 0)
#define _mm_setyyyy_ps(v) vdup_lane_f32(v, 1)
#define _mm_setzzzz_ps(v) vdup_lane_f32(v, 2)
#define _mm_setwwww_ps(v) vdup_lane_f32(v, 3)

#define _mm_setzero_ps() _mm_set1_ps(0.0f)

// mac/ios intrinsics use direct access (what about on _m128i?)
// should use rr[ee], but that requires operator[] calls to be inlined
#define macroVec4Elem(rr, ee) *( ((float*)&rr) + (ee))

finline float32x4_t _mm_and_ps(const float32x4_t& a, const float32x4_t& b) 
{ 
    return vreinterpretq_f32_u32(vandq_s32(vreinterpretq_u32_s32(a),vreinterpretq_u32_s32(b))); 
} 

finline float32x4_t _mm_cmpneq_ps(const float32x4_t& a, const float32x4_t& b) { 
    return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a,b))); 
} 

// a = x.x,a.y,a.z,a.w
finline float32x4_t _mm_setx_ps(const float32x4_t& aa, const float32x4_t& xx)
{
    // TODO: this is supposed to work, but it barks about shadowed local variable
    //return vsetq_lane_f32(vgetq_lane_f32(xx, 0), aa, 0);
    
    float32x4_t aCopy = aa;
    macroVec4Elem(aCopy, 0) = macroVec4Elem(xx, 0);
    return aCopy;
}

#define _mm_mul_ps(a,b) vmulq_f32(a,b)

#define _mm_fixzero_ps(a,b) _mm_and_ps(a, _mm_cmpneq_ps(b, _mm_setzero_ps() ))
                          
// rqrt (high precision)
finline float32x4_t _mm_rsqrthp_ps(const float32x4_t& a) 
{ 
    float32x4_t est  = vrsqrteq_f32(a);
    
    est = _mm_fixzero_ps(est, a);
    
    // newton raphson
    float32x4_t stepA = vrsqrtsq_f32(a, vmulq_f32(est,est));  // xn+1 = xn(3-dxn*dxn)/2
    
    return _mm_mul_ps(est, stepA);
}

// sqrt
finline float32x4_t _mm_sqrthp_ps(const float32x4_t& a) 
{ 
    // sqrt(a) = a * rsqrt(a) 
    return _mm_mul_ps(_mm_rsqrthp_ps(a), a);
}

// recip
finline float32x4_t _mm_rcphp_ps(const float32x4_t& a)
{ 
    float32x4_t est  = vrecpeq_f32(a);
    
    est = _mm_fixzero_ps(est, a);
    
    float32x4_t stepA = vrecpsq_f32(est, a);  // xn+1 = xn(2-dxn) 
    
    return _mm_mul_ps(est, stepA);
}

// this is doing 4 ops, could cut to 2 ops (no single ops like ss)
//  implemented that, but it took more ops (esp fixzero)
#define _mm_rsqrthp_ss(a) _mm_setx_ps(a, _mm_rsqrthp_ps(a))                  
#define _mm_sqrthp_ss(a)  _mm_setx_ps(a, _mm_sqrthp_ps(a))
#define _mm_rcphp_ss(a)   _mm_setx_ps(a, _mm_rcphp_ps(a))

#endif