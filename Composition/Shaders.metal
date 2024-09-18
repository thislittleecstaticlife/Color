//
//  Shaders.metal
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

#include <Graphics/Jzazbz.hpp>
#include <metal_stdlib>

using namespace metal;

//===------------------------------------------------------------------------===
// • Local functions
//===------------------------------------------------------------------------===

namespace
{
    float3 find_max_chroma_color(float lane_t, float height, float hue)
    {
        // • Find the Display P3 max chroma edge for the given hue
        //
        const auto edges = jzazbz::find_max_chroma_edge_P3(hue);

        // • Perform binary search along edge to find the target hue
        //
        auto lower = edges.lower;
        auto upper = edges.upper;

        for (auto i = 0; i < 4; i++)
        {
            const auto val      = mix(lower, upper, lane_t);
            const auto jab      = jzazbz::from_LMS(val.xyz);
            const auto test_hue = atan2(jab[2], jab[1]);

            const auto test      = (test_hue <= hue) ? lane_t : -1.0;
            const auto low_t     = max(0.0f, simd_max(test));
            const auto new_lower = mix(lower, upper, low_t);
            const auto new_upper = mix(lower, upper, low_t + 1.0f/height);

            lower = new_lower;
            upper = new_upper;
        }

        return jzazbz::from_LMS(lower.xyz);
    }

} // namespace <anonymout>

//===------------------------------------------------------------------------===
// Generate hue gradient
//===------------------------------------------------------------------------===

[[kernel]] void generate_hue_gradient
(
    texture1d<half,access::write> output    [[ texture(0)              ]],
    ushort2                       gid       [[ thread_position_in_grid ]],
    ushort2                       grid_size [[ threads_per_grid        ]]
)
{
    const auto height    = static_cast<float>(grid_size.x);
    const auto lane_t    = static_cast<float>(gid.x) / height;
    const auto hue_t     = (static_cast<float>(gid.y) + 0.5f) / static_cast<float>(grid_size.y);
    const auto hue       = mix(-M_PI_F, M_PI_F, hue_t);
    const auto max_c_jab = find_max_chroma_color(lane_t, height, hue);

    if (0 == gid.x)
    {
        const auto lrgb = half3( jzazbz::convert_to_linear_display_P3(max_c_jab) );

        output.write( half4(lrgb, 1.0h), gid.y );
    }
}

//===------------------------------------------------------------------------===
// HueGradientVertex
//===------------------------------------------------------------------------===

struct HueGradientVertex
{
    float4 position [[ position ]];
    float  tex_coord;
};

//===------------------------------------------------------------------------===
// Hue gradient texture
//===------------------------------------------------------------------------===

[[fragment]] half4 hue_gradient_fragment(HueGradientVertex               in      [[ stage_in  ]],
                                         texture1d<half, access::sample> texture [[ texture(0)]])
{
    constexpr auto s = sampler{ coord::normalized, address::repeat, filter::linear };

    return texture.sample(s, in.tex_coord);
}

/*
[[vertex]] HueGradientVertex hue_gradient_vertex(constant State& state [[ buffer(0) ]],
                                                 unsigned int    vid   [[ vertex_id ]])
{
    // Uniform position
    const auto xu = (0 == (vid & 0x02)) ? 0.0f : 1.0f;

    // Device position in lower right
    const auto xd = (0 == (vid & 0x02)) ? state.hue_gradient_rect.left   : state.hue_gradient_rect.right;
    const auto yd = (0 == (vid & 0x01)) ? state.hue_gradient_rect.bottom : state.hue_gradient_rect.top;

    // Normalized position
    const auto xn = fma(xd, 2.0f/state.view_size.x, -1.0f);
    const auto yn = fma(yd, 2.0f/state.view_size.y, -1.0f);

    return {
        .position  = float4{ xn, yn, 0.7f, 1.0f },
        .tex_coord = fma( fmod(state.hue, 360.0f), 1.0f/360.0f, xu - 0.5f )
    };
}
*/

//===------------------------------------------------------------------------===
// • Generate_vertices
//===------------------------------------------------------------------------===

[[kernel]] void generate_vertices(constant float& hue       [[ buffer(0)               ]],
                                  device float4*  output    [[ buffer(1)               ]],
                                  ushort2         gid       [[ thread_position_in_grid ]],
                                  ushort2         grid_size [[ threads_per_grid        ]])
{
    // • gid.x is simd_lane_id and grid_size.x is threads_per_simdgroup (= thread_execution_width)

    // • Find the Jzazbz value with the highest chroma at `hue` - each simd group performs the
    //   same search instead of using a shared computation (for now)
    //
    const auto height     = static_cast<float>(grid_size.x);
    const auto lane_t     = static_cast<float>(gid.x) / height;
    const auto target_hue = (hue < 180.0f ? hue : hue - 360.0f) * M_PI_F / 180.f;
    const auto max_c_jab  = find_max_chroma_color(lane_t, height, target_hue);

    // • Find the top and bottom intersections with the in-gamut Jzazbz solid at the current hue
    //
    const auto width  = static_cast<float>(grid_size.y);
    const auto band_t = static_cast<float>(gid.y) / width;

    auto outer_high = mix({ 0.16717463103478347f *  1.33f, 0.0f, 0.0f }, max_c_jab, band_t);
    auto inner_high = mix({ 0.16717463103478347f *  0.67f, 0.0f, 0.0f }, max_c_jab, band_t);
    auto inner_low  = mix({ 0.16717463103478347f *  0.25f, 0.0f, 0.0f }, max_c_jab, band_t);
    auto outer_low  = mix({ 0.16717463103478347f * -0.67f, 0.0f, 0.0f }, max_c_jab, band_t);

    constexpr auto max_lrgb = float3(1.0f);
    constexpr auto min_lrgb = float3(0.0f);

    for (auto i = 0; i < 4; i++)
    {
        const auto upper_jab     = mix(inner_high, outer_high, lane_t);
        const auto upper_lrgb    = jzazbz::convert_to_linear_display_P3(upper_jab);
        const auto upper_clamped = clamp(upper_lrgb, min_lrgb, max_lrgb);

        const auto lower_jab     = mix(inner_low, outer_low, lane_t);
        const auto lower_lrgb    = jzazbz::convert_to_linear_display_P3(lower_jab);
        const auto lower_clamped = clamp(lower_lrgb, min_lrgb, max_lrgb);

        const auto upper_test = ( all(upper_lrgb == upper_clamped) ) ? lane_t : -1.0f;

        const auto lower_test = ( 0.0f <= lower_jab.x && all(lower_lrgb == lower_clamped) )
                                ? lane_t
                                : -1.0f;

        const auto upper_t = simd_max(upper_test);
        const auto lower_t = simd_max(lower_test);

        const auto new_outer_high = mix(inner_high, outer_high, upper_t + 1.0f/height);
        const auto new_inner_high = mix(inner_high, outer_high, upper_t);
        const auto new_inner_low  = mix(inner_low, outer_low, lower_t);
        const auto new_outer_low  = mix(inner_low, outer_low, lower_t + 1.0f/height);

        outer_high = new_outer_high;
        inner_high = new_inner_high;
        inner_low  = new_inner_low;
        outer_low  = new_outer_low;
    }

    output[2*gid.y]   = float4( inner_low,  0.0f );
    output[2*gid.y+1] = float4( inner_high, 0.0f );

    if (gid.y+1 == grid_size.y)
    {
        output[2*gid.y+2] = float4( max_c_jab, 0.0f );
    }
}

//===------------------------------------------------------------------------===
// • VertexOut
//===------------------------------------------------------------------------===

struct VertexOut
{
    float4 position [[ position ]];
    float4 color;
};

//===------------------------------------------------------------------------===
// • Foreground
//===------------------------------------------------------------------------===

[[fragment]] half4 foreground_fragment(VertexOut input [[ stage_in  ]])
{
    const auto lrgb = jzazbz::convert_to_linear_display_P3(input.color.xyz);

    return half4( half3(lrgb), 1.0h );
}

[[vertex]] VertexOut foreground_vertex(const device float4* vertices [[ buffer(0) ]],
                                       uint                 vid      [[ vertex_id ]])
{
    const auto v = vertices[vid];
    const auto y = v.x / 0.16717463103478347f;
    const auto C = length(v.yz) / 0.17535403f;

    return {
        .position = float4{ C*2.0f - 1.0f, y*2.0f - 1.0f, 0.5f, 1.0f },
        .color    = v
    };
}

//===------------------------------------------------------------------------===
// • Background
//===------------------------------------------------------------------------===

[[fragment]] half4 background_fragment(VertexOut input [[ stage_in ]])
{
    // • input.color[3] is unit y position. At (1, 0) in unit coordinates chroma is 0.024;
    //  at (1, 1), 0.058. At unit x = 0 for all y values, chroma is cmin,
    //  so C = cmin + (mix(0.024, 0.058, yu) - cmin) * xu,
    constexpr auto Cmin = 0.024f/3.0f;

    const auto Cd = mix(0.024f, 0.058f, input.color[3]) - Cmin;
    const auto a  = Cmin + (Cd * input.color[1]);
    const auto b  = Cmin + (Cd * input.color[2]);
    const auto J  = input.color[0];

    const auto lrgb = jzazbz::convert_to_linear_display_P3({ J, a, b });

    return half4( half3(lrgb), 1.0h );
}

[[vertex]] VertexOut background_vertex(constant float& hue [[ buffer(0) ]],
                                       uint            vid [[ vertex_id ]])
{
    // • Clockwise quad triangle strip
    //
    //  1   3
    //  | \ |
    //  0   2
    //
    const auto is_left = 0 != (vid & 0b10);
    const auto is_top  = 0 != (vid & 0b01);

    // • Normalized x and y coordinates
    //
    const auto xn = is_left ? -1.0f :  1.0f;
    const auto yn = is_top  ?  1.0f : -1.0f;

    // • Unit [0, 1]
    //
    const auto xu = is_left ? 0.0f : 1.0f;
    const auto yu = is_top  ? 1.0f : 0.0f;

    // • Color from coordinates
    //
    const auto Jz  = mix(0.032608401221558024f, 0.12133886641726202f, yu);
    const auto caz = cospi(hue / 180.0f) * xu; // az chroma scale
    const auto cbz = sinpi(hue / 180.0f) * xu; // bz chroma scale

    return {
        .position = float4( xn, yn, 0.75f, 1.0f ),
        .color    = float4{ Jz, caz, cbz, yu }
    };
}
