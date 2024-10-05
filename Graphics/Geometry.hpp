//
//  Geometry.hpp
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

#include <Data/Layout.hpp>
#include <simd/simd.h>

#if !defined ( __METAL_VERSION__ )
#include <type_traits>
#endif

//===------------------------------------------------------------------------===
// • namespace geometry
//===------------------------------------------------------------------------===

namespace geometry
{

//===------------------------------------------------------------------------===
// • Region
//===------------------------------------------------------------------------===

struct Region
{
    uint32_t    left;
    uint32_t    top;
    uint32_t    right;
    uint32_t    bottom;
};

static_assert( 16 ==  sizeof(Region), "Unexpected size" );
static_assert(  4 == alignof(Region), "Unexpected alignment" );

#if !defined ( __METAL_VERSION__ )
static_assert( data::is_trivial_layout<Region>(), "Unexpected layout" );
#endif

//===------------------------------------------------------------------------===
// • Region Utilities
//===------------------------------------------------------------------------===

constexpr uint32_t width(const Region rgn)
{
    return rgn.right - rgn.left;
}

constexpr uint32_t height(const Region rgn)
{
    return rgn.bottom - rgn.top;
}

constexpr simd::uint2 origin(const Region rgn)
{
    return { rgn.left, rgn.top };
}

constexpr simd::uint2 size(const Region rgn)
{
    return { width(rgn), height(rgn) };
}

constexpr bool contains(const Region rgn, simd::uint2 point)
{
    return rgn.left <= point.x && point.x < rgn.right
        && rgn.top  <= point.y && point.y < rgn.bottom;
}

constexpr bool operator == (const Region lhs, const Region rhs)
{
    return lhs.left   == rhs.left
        && lhs.top    == rhs.top
        && lhs.right  == rhs.right
        && lhs.bottom == rhs.bottom;
}

constexpr Region operator + (const Region rgn, simd::int2 offset)
{
    return {
        .left   = rgn.left   + offset.x,
        .top    = rgn.top    + offset.y,
        .right  = rgn.right  + offset.x,
        .bottom = rgn.bottom + offset.y
    };
}

#if !defined ( __METAL_VERSION__ )
inline Region& operator += (Region& rgn, simd::int2 offset)
{
    rgn.left   += offset.x;
    rgn.top    += offset.y;
    rgn.right  += offset.x;
    rgn.bottom += offset.y;

    return rgn;
}
#endif

//===------------------------------------------------------------------------===
// Rectangle
//===------------------------------------------------------------------------===

struct Rectangle
{
    float   left;
    float   top;
    float   right;
    float   bottom;
};

static_assert( 16 ==  sizeof(Rectangle), "Unexpected size" );
static_assert(  4 == alignof(Rectangle), "Unexpected alignment" );

#if !defined ( __METAL_VERSION__ )
static_assert( data::is_trivial_layout<Rectangle>(), "Unexpected layout" );
#endif

//===------------------------------------------------------------------------===
// • Rectangle Utilities
//===------------------------------------------------------------------------===

constexpr float width(const Rectangle rect)
{
    return rect.right - rect.left;
}

constexpr float height(const Rectangle rect)
{
    return rect.bottom - rect.top;
}

constexpr simd::float2 origin(const Rectangle rect)
{
    return { rect.left, rect.top };
}

constexpr simd::float2 size(const Rectangle rect)
{
    return { width(rect), height(rect) };
}

constexpr float center_x(const Rectangle rect)
{
    return rect.left + 0.5f*width(rect);
}

constexpr float center_y(const Rectangle rect)
{
    return rect.top + 0.5f*height(rect);
}

constexpr simd::float2 center(const Rectangle rect)
{
    return origin(rect) + 0.5f*size(rect);
}

constexpr bool operator == (const Rectangle lhs, const Rectangle rhs)
{
    return lhs.left   == rhs.left
        && lhs.top    == rhs.top
        && lhs.right  == rhs.right
        && lhs.bottom == rhs.bottom;
}

//===------------------------------------------------------------------------===
// TextureRect
//===------------------------------------------------------------------------===

struct TextureRect
{
    float   left;
    float   top;
    float   right;
    float   bottom;
};

static_assert( 16 ==  sizeof(TextureRect), "Unexpected size" );
static_assert(  4 == alignof(TextureRect), "Unexpected alignment" );

#if !defined ( __METAL_VERSION__ )
static_assert( data::is_trivial_layout<TextureRect>(), "Unexpected layout" );
#endif

//===------------------------------------------------------------------------===
// • TextureRect Utilities
//===------------------------------------------------------------------------===

constexpr float width(const TextureRect rect)
{
    return rect.right - rect.left;
}

constexpr float height(const TextureRect rect)
{
    return rect.bottom - rect.top;
}

constexpr simd::float2 origin(const TextureRect rect)
{
    return { rect.left, rect.top };
}

constexpr simd::float2 size(const TextureRect rect)
{
    return { width(rect), height(rect) };
}

constexpr float center_x(const TextureRect rect)
{
    return rect.left + 0.5f*width(rect);
}

constexpr float center_y(const TextureRect rect)
{
    return rect.top + 0.5f*height(rect);
}

constexpr simd::float2 center(const TextureRect rect)
{
    return origin(rect) + 0.5f*size(rect);
}

constexpr bool operator == (const TextureRect lhs, const TextureRect rhs)
{
    return lhs.left   == rhs.left
        && lhs.top    == rhs.top
        && lhs.right  == rhs.right
        && lhs.bottom == rhs.bottom;
}

//===------------------------------------------------------------------------===
// DeviceRect
//===------------------------------------------------------------------------===

struct DeviceRect
{
    float   left;
    float   top;
    float   right;
    float   bottom;
};

static_assert( 16 ==  sizeof(DeviceRect), "Unexpected size" );
static_assert(  4 == alignof(DeviceRect), "Unexpected alignment" );

#if !defined ( __METAL_VERSION__ )
static_assert( data::is_trivial_layout<DeviceRect>(), "Unexpected layout" );
#endif

//===------------------------------------------------------------------------===
// • DeviceRect Utilities
//===------------------------------------------------------------------------===

constexpr float width(const DeviceRect rect)
{
    return rect.right - rect.left;
}

constexpr float height(const DeviceRect rect)
{
    return rect.top - rect.bottom;
}

constexpr simd::float2 origin(const DeviceRect rect)
{
    return { rect.left, rect.top };
}

constexpr simd::float2 size(const DeviceRect rect)
{
    return { width(rect), height(rect) };
}

constexpr float center_x(const DeviceRect rect)
{
    return rect.left + 0.5f*width(rect);
}

constexpr float center_y(const DeviceRect rect)
{
    return rect.bottom + 0.5f*height(rect);
}

constexpr simd::float2 center(const DeviceRect rect)
{
    return origin(rect) + 0.5f*size(rect);
}

constexpr bool operator == (const DeviceRect lhs, const DeviceRect rhs)
{
    return lhs.left   == rhs.left
        && lhs.top    == rhs.top
        && lhs.right  == rhs.right
        && lhs.bottom == rhs.bottom;
}

//===------------------------------------------------------------------------===
// • simd type conversion
//===------------------------------------------------------------------------===

constexpr simd::float2 make_float2(simd::uint2 source)
{
    return { static_cast<float>(source.x), static_cast<float>(source.y) };
}

//===------------------------------------------------------------------------===
//
// • Initialization
//
//===------------------------------------------------------------------------===

//===------------------------------------------------------------------------===
// • Region
//===------------------------------------------------------------------------===

constexpr Region make_region_of_size(simd::uint2 size)
{
    return {
        .left   = 0u,
        .top    = 0u,
        .right  = size.x,
        .bottom = size.y
    };
}

constexpr Region make_region(simd::uint2 origin, simd::uint2 size)
{
    return {
        .left   = origin.x,
        .top    = origin.y,
        .right  = origin.x + size.x,
        .bottom = origin.y + size.y
    };
}

//===------------------------------------------------------------------------===
// • Rectangle
//===------------------------------------------------------------------------===

constexpr Rectangle make_rectangle_of_size(simd::float2 size)
{
    return { .left = 0.0f, .top = 0.0f, .right = size.x, .bottom = size.y };
}

constexpr Rectangle make_rectangle_of_size(simd::uint2 size)
{
    return make_rectangle_of_size( make_float2(size) );
}

constexpr Rectangle center_rectangle(const Rectangle rect, const Rectangle bounds)
{
    const auto origin = center(bounds) - 0.5f*size(rect);

    return {
        .left   = origin.x,
        .top    = origin.y,
        .right  = origin.x + width(rect),
        .bottom = origin.y + height(rect)
    };
}

//===------------------------------------------------------------------------===
//
// • Conversion
//
//===------------------------------------------------------------------------===

//===------------------------------------------------------------------------===
// • Rectangle
//===------------------------------------------------------------------------===

constexpr Rectangle make_rectangle(const Region rgn)
{
    return {
        .left   = static_cast<float>(rgn.left),
        .top    = static_cast<float>(rgn.top),
        .right  = static_cast<float>(rgn.right),
        .bottom = static_cast<float>(rgn.bottom)
    };
}

constexpr Rectangle make_rectangle(const TextureRect tr, simd::float2 size)
{
    return {
        .left   = tr.left   * size.x,
        .top    = tr.top    * size.y,
        .right  = tr.right  * size.x,
        .bottom = tr.bottom * size.y
    };
}

constexpr Rectangle make_rectangle(const TextureRect tr, simd::uint2 size)
{
    return make_rectangle( tr, make_float2(size) );
}

constexpr Rectangle make_rectangle(const DeviceRect dr, simd::float2 size)
{
    return {
        .left   = 0.5f * size.x * (dr.left + 1.0f),
        .top    = 0.5f * size.y * (1.0f - dr.top),
        .right  = 0.5f * size.x * (dr.right + 1.0f),
        .bottom = 0.5f * size.y * (1.0f - dr.bottom)
    };
}

constexpr Rectangle make_rectangle(const DeviceRect dr, simd::uint2 size)
{
    return make_rectangle( dr, make_float2(size) );
}

//===------------------------------------------------------------------------===
// • TextureRect
//===------------------------------------------------------------------------===

constexpr TextureRect full_texture_rect(void)
{
    return { .left = 0.0f, .top = 0.0f, .right = 1.0f, .bottom = 1.0f };
}

constexpr TextureRect make_texture_rect(const Region rgn, simd::uint2 size)
{
    return {
        .left   = static_cast<float>(rgn.left)   / static_cast<float>(size.x),
        .top    = static_cast<float>(rgn.top)    / static_cast<float>(size.y),
        .right  = static_cast<float>(rgn.right)  / static_cast<float>(size.x),
        .bottom = static_cast<float>(rgn.bottom) / static_cast<float>(size.y)
    };
}

constexpr TextureRect make_texture_rect(const Rectangle rect, simd::float2 size)
{
    return {
        .left   = rect.left   / size.x,
        .top    = rect.top    / size.y,
        .right  = rect.right  / size.x,
        .bottom = rect.bottom / size.y
    };
}

constexpr TextureRect make_texture_rect(const Rectangle rect, simd::uint2 size)
{
    return make_texture_rect( rect, make_float2(size) );
}

constexpr TextureRect make_texture_rect(const DeviceRect dr)
{
    return {
        .left   = 0.5f * (dr.left + 1.0f),
        .top    = 0.5f * (1.0f - dr.top),
        .right  = 0.5f * (dr.right + 1.0f),
        .bottom = 0.5f * (1.0f - dr.bottom)
    };
}

//===------------------------------------------------------------------------===
// • DeviceRect
//===------------------------------------------------------------------------===

constexpr DeviceRect full_device_rect(void)
{
    return { .left = -1.0f, .top = 1.0f, .right = 1.0f, .bottom = -1.0f };
}

constexpr DeviceRect make_device_rect(const Region rgn, simd::uint2 size)
{
    return {
        .left   = -1.0f + 2.0f*static_cast<float>(rgn.left)   / static_cast<float>(size.x),
        .top    =  1.0f - 2.0f*static_cast<float>(rgn.top)    / static_cast<float>(size.y),
        .right  = -1.0f + 2.0f*static_cast<float>(rgn.right)  / static_cast<float>(size.x),
        .bottom =  1.0f - 2.0f*static_cast<float>(rgn.bottom) / static_cast<float>(size.y)
    };
}

constexpr DeviceRect make_device_rect(const Rectangle rect, simd::float2 size)
{
    return {
        .left   = -1.0f + 2.0f*rect.left   / size.x,
        .top    =  1.0f - 2.0f*rect.top    / size.y,
        .right  = -1.0f + 2.0f*rect.right  / size.x,
        .bottom =  1.0f - 2.0f*rect.bottom / size.y
    };
}

constexpr DeviceRect make_device_rect(const Rectangle rect, simd::uint2 size)
{
    return make_device_rect(rect, make_float2(size) );
}

constexpr DeviceRect make_device_rect(const TextureRect tr)
{
    return {
        .left   = -1.0f + 2.0f*tr.left,
        .top    =  1.0f - 2.0f*tr.top,
        .right  = -1.0f + 2.0f*tr.right,
        .bottom =  1.0f - 2.0f*tr.bottom
    };
}

//===------------------------------------------------------------------------===
// • Size to fit
//===------------------------------------------------------------------------===

constexpr Rectangle size_to_fit(simd::float2 aspect, Rectangle rect)
{
    const auto fit_scale = size(rect) / aspect;

    if (fit_scale.x < fit_scale.y)
    {
        // • Constrained in width
        //
        const auto height = aspect.y * fit_scale.x;

        rect.top    = center_y(rect) - 0.5f*height;
        rect.bottom = rect.top + height;
    }
    else
    {
        // • Constrained in height
        //
        const auto width = aspect.x * fit_scale.y;

        rect.left  = center_x(rect) - 0.5f*width;
        rect.right = rect.left + width;
    }

    return rect;
}

constexpr Rectangle size_to_fit(simd::uint2 aspect, Rectangle rect)
{
    return size_to_fit( make_float2(aspect), rect );
}

} // namespace geometry
