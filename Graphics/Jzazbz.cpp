//
//  Jzazbz.cpp
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

//===------------------------------------------------------------------------===
// • namespace jzazbz
//===------------------------------------------------------------------------===

namespace jzazbz
{

//===------------------------------------------------------------------------===
// • find_max_chroma_color
//===------------------------------------------------------------------------===

simd::float3 find_max_chroma_color(float hue)
{
    // • Find the Display P3 max chroma edge for the given hue
    //
    const auto target_hue     = (hue < 180.0f) ? hue : hue - 360.0f;
    const auto target_radians = target_hue * M_PI / 180.0f;
    const auto edges          = jzazbz::find_max_chroma_edge_P3(target_hue);

    // • Perform binary search along edge to find the target hue
    //
    auto lower = edges.lower;
    auto upper = edges.upper;

    for (auto i = 0; i < 20; i++)
    {
        const auto val      = lower + 0.5f*(upper - lower);
        const auto jab      = jzazbz::from_LMS(val.xyz);
        const auto test_hue = atan2(jab[2], jab[1]);

        if (test_hue <= target_radians)
        {
            // Inside RGB gamut
            lower = val;
        }
        else
        {
            // Outside RGB gamut
            upper = val;
        }
    }

    return jzazbz::from_LMS(lower.xyz);
}

} // namespace jzazbz
