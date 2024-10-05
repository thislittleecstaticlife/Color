//
//  CompositionData.hpp
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

#include <Graphics/Geometry.hpp>
#include <simd/simd.h>

//===------------------------------------------------------------------------===
//
// • CompositionData
//
//===------------------------------------------------------------------------===

struct CompositionData
{
    simd::uint2         grid_size;
    geometry::Region    jc_region;
    geometry::Region    gradient_region;
    geometry::Region    max_c_region;

    float               hue;
    simd::float3        max_c_color;
};

#if !defined ( __METAL_VERSION__ )
static_assert( data::is_trivial_layout<CompositionData>(), "Unexpected layout" );
#endif
