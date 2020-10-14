# sseneonmath
MIT License.

High-precision ops for rcp, rsqrt, and sqrt for the SSE and Neon SIMD instruction sets.  These are designated with "hp" like _mm_rsqrthp_ps.  Neon is a great superset of SSE without the missing ops.  Also Neon doesn't as many compilation flags or variants.  SSE has moved from 16-byte to 32- and 64-byte ops.  I typically target AVX1, but write to SSE2 level ops.  Later AVX and SSE3/4 intrinsics need a lot more fallback code to avoid crashing on older Intel processors.  Neon should mostly just work going back many generations of CPUs.

The vecmath.cpp file is part of a SIMD abstraction layer that I wrote for a math library that works with SSE2 and Neon.  There are now packages like the Accelerate vector math library from Apple and DirectXMath library Microsoft but back then these weren't available.  You may find better versions of the following routines but those are much bigger libraries.  Accelerate has fast and precise versions of calls that also translate to Metal.  SSE2Neon also maps SSE intrinsics to Neon and then you keep one set of code.

Since Intel's SSE intrinsics supply a high-precision of mm_sqrt_ps, developers often assume mm_rsqrt_ps and mm_rcp_ps are also high precision.  But these routines need Newton-Raphson (NR) iteration for reasonably precise results.  The precision doubles from 10-/12-bit to 20-/24-bit in just one iteration.  This should be high enough for fp32 calculations.  I mostly uses this for normalizing vectors.

ARM Neon has a few gotchas.  Neon originally lacked the mm_shuffle_ps op found in SSE, but newer Neon instructions cover that.  The rcp/rsqrt intrinsic ops also simplify using NR, and so the ops are slightly different there.  I don't know about bit-for-bit results across these calls.

I wrote these routines over a decade ago, and so there may be better instructions to use in more recent releases of Neon or SSE3+.  Caution must also be taken with NaN and INF handling, and these add to the instruction counts, or live dangerously for higher performance.   These don't have an extensive test suite, but I used these for rendering.  Releasing these under MIT so that more libraries run with SIMD (and hopefully port to the GPU).

Some vector approximations to use with these:
div(a,b) = _mm_mul_ps(a, _mm_rcphp_ps(b))   // a * rcp(b) 
sqrt(a)  = _mm_mul_ps(a, _mm_sqrthp_ps(a))  // a * rsqrt(a)

Here are links to some of those other SIMD packages.
https://github.com/jratcliff63367/sse2neon/blob/master/SSE2NEON.h
https://docs.microsoft.com/en-us/windows/win32/dxmath/directxmath-portal
https://developer.apple.com/documentation/accelerate/working_with_vectors

To compile these, I used the following commands:

macOS:
clang++ -c vecmath.cpp 

iOS:
clang++ -arch arm64e -mios-version-min=14.0 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS14.0.sdk vecmath.cpp





