//
//  Jzazbz.hpp
//
//  Copyright © 2024 Robert Guequierre
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

#pragma once

#if defined ( __METAL_VERSION__ )
#include <metal_stdlib>
#endif

#include <simd/simd.h>

//===------------------------------------------------------------------------===
//
// • Jzazbz to Linear RGB Conversion
//
//===------------------------------------------------------------------------===

namespace jzazbz
{

//===------------------------------------------------------------------------===
// • Jzazbz to LMS
//===------------------------------------------------------------------------===

inline simd::float3 convert_to_LMS(simd::float3 jab)
{
    const auto M_IzazbzToLMSp = simd::float3x3 {
        simd::float3{ 1.0f,                 1.0f,                 1.0f                },
        simd::float3{ 0.138605043271539f,  -0.138605043271539f,  -0.0960192420263189f },
        simd::float3{ 0.0580473161561189f, -0.0580473161561189f, -0.811891896056039f  }
    };

    constexpr auto d     = -0.56f;
    constexpr auto d0    =  1.6295499532821566e-11f;

    constexpr auto vc1   = simd::float3( 3424.0f/4096.0f );
    constexpr auto vc2   = simd::float3( 2413.0f/128.0f );
    constexpr auto vc3   = 2392.0f/128.0f;
    constexpr auto vInvP = 32.0f / (1.7f * 2523.0f);
    constexpr auto vInvN = 16384.0f / 2610.0f;

    // actually 0.000000000037035, adjusted for precision limits
    constexpr auto minLMSp = simd::float3(0.0000000000370353f);
    constexpr auto maxLMSp = simd::float3(3.227f);

    const auto Jzp    = jab[0] + d0;
    const auto Iz     = Jzp / (1.0f + d - d*Jzp);
    const auto LMSp   = M_IzazbzToLMSp * simd::float3{ Iz, jab[1], jab[2] };
    const auto LMSpc  = simd::clamp(LMSp, minLMSp, maxLMSp);

#if !defined ( __METAL_VERSION__ )
    const auto LMSpp1 = simd::pow( LMSpc, simd::float3(vInvP) );
    const auto LMSpp2 = (vc1 - LMSpp1) / (vc3*LMSpp1 - vc2);
    const auto LMS    = 100.0f * simd::pow( LMSpp2, simd::float3(vInvN) );
#else
    const auto LMSpp1 = metal::powr(LMSpc, vInvP);
    const auto LMSpp2 = (vc1 - LMSpp1) / (vc3*LMSpp1 - vc2);
    const auto LMS    = 100.0f * metal::powr(LMSpp2, vInvN);
#endif

    return LMS;
}

//===------------------------------------------------------------------------===
// • Covnersion to Linear Display P3
//===------------------------------------------------------------------------===

inline simd::float3 LMS_to_linear_display_P3(simd::float3 lms)
{
    // M_XYZToLinearP3 = [  2.49350912393461  -0.829473213929555   0.035851264433918  ] T
    //                   [ -0.931388179404779  1.7626305796003    -0.0761839369220758 ]
    //                   [ -0.402712756741652  0.0236242371055886  0.957029586694311  ]

    // M_LMSToLinearP3 = M_XYZToLinearP3 * M_XYZpToXYZD65 * M_LMSToXYZD65p
    const auto M_LMSToLinearP3 = simd::float3x3 {
        simd::float3{  4.4820606379518333f,  -1.9532025238860451f,  -0.0027453573623004834f },
        simd::float3{ -3.6184317541411817f,   3.5217700975984596f,  -0.45182653146288487f   },
        simd::float3{  0.16694496856407345f, -0.54063532522070301f,  1.4822547119502889f    },
    };

    return M_LMSToLinearP3 * lms;
}

inline simd::float3 convert_to_linear_display_P3(simd::float3 jab)
{
    return LMS_to_linear_display_P3( convert_to_LMS(jab) );
}

//===------------------------------------------------------------------------===
// • Jzazbz from LMS
//===------------------------------------------------------------------------===

inline simd::float3 from_LMS(simd::float3 lms)
{
    // 0.5       0.5       0
    // 3.524000 -4.066708  0.542708
    // 0.199076  1.096799 -1.295875
    const auto M_LMSpToIzazbz = simd::float3x3{
        simd::float3{ 0.5f,  3.524000f,  0.199076f },
        simd::float3{ 0.5f, -4.066708f,  1.096799f },
        simd::float3{ 0.0f,  0.542708f, -1.295875f }
    };

    constexpr auto c1 = simd::float3( 3424.0f / 4096.0f );
    constexpr auto c2 = 2413.0f / 128.0f;
    constexpr auto c3 = 2392.0f / 128.0f;
    constexpr auto n  = 2610.0f / 16384.0f;
    constexpr auto p  = 1.7f * 2523.0f / 32.0f;

    constexpr auto d  = -0.56f;
    constexpr auto d0 =  1.6295499532821566e-11f;

#if !defined ( __METAL_VERSION__ )
    const auto valp     = simd::pow( simd::max(lms/100.0f, simd::float3(0.0f)), simd::float3(n) );
    const auto fraction = (c1 + c2*valp) / (simd::float3(1.0f) + c3*valp);
    const auto lmsp     = simd::pow( fraction, simd::float3(p) );
#else
    const auto valp     = metal::powr( simd::max(lms/100.0f, 0.0f), n );
    const auto fraction = (c1 + c2*valp) / (simd::float3(1.0f) + c3*valp);
    const auto lmsp     = metal::powr(fraction, p);
#endif

    const auto Izazbz   = M_LMSpToIzazbz * lmsp;
    const auto Jzn      = (1.0f + d) * Izazbz[0];
    const auto Jzd      =  1.0f + d*Izazbz[0];
    const auto Jz       = Jzn / Jzd - d0;

    return { Jz, Izazbz[1], Izazbz[2] };
}

//===------------------------------------------------------------------------===
// • Max-chroma edge
//===------------------------------------------------------------------------===

struct LMSChromaEdge
{
    simd::float4    lower;
    simd::float4    upper;
};

inline LMSChromaEdge find_max_chroma_edge_P3(float hue)
{
    // • Primary colors in LMS (from Display P3)
    //
    const simd::float4 corners[] = {
        { 0.5160874353648806f,  0.6689515188836437f,  0.6434469935994587f,   -M_PI_F               },
        { 0.55608700197488292f, 0.73025516799564405f, 0.89827700087481577f,  -2.7604618631505451f  }, // cyan
        { 0.11431238432553269f, 0.17519605565166838f, 0.72826353378675235f,  -1.7688992503294745f  }, // blue
        { 0.53001160774764933f, 0.41718828256028762f, 0.8027984639562511f,   -0.60623058828496412f }, // magenta
        { 0.41569922342211668f, 0.24199222690861924f, 0.074534930169498803f,  0.74690126898001996f }, // red
        { 0.85747384107146684f, 0.79705133925259486f, 0.24454839725756228f,   1.789331917784555f   }, // yellow
        { 0.44177461764935022f, 0.55505911234397565f, 0.17001346708806347f,   2.3782967581439904f  }, // green
        { 0.5160874353648806f,  0.6689515188836437f,  0.6434469935994587f,    M_PI_F               },
    };

    // • Find the edge along which the hue lies
    //
    auto j = 0;
    j += (corners[4].w   <= hue) ? 4 : 0;
    j += (corners[j+2].w <= hue) ? 2 : 0;
    j += (corners[j+1].w <= hue) ? 1 : 0;

    return  {
        .lower = corners[j],
        .upper = corners[j+1]
    };
}

} // namespace jzazbz
