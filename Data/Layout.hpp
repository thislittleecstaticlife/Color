//
//  Layout.hpp
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

#if !defined ( __METAL_VERSION__ )
#include <type_traits>
#endif

//===------------------------------------------------------------------------===
// • namespace data
//===------------------------------------------------------------------------===

namespace data
{

#if !defined ( __METAL_VERSION__ )

//===------------------------------------------------------------------------===
//
// • Data layout (Host only)
//
//===------------------------------------------------------------------------===

//===------------------------------------------------------------------------===
// • TrivialLayout concept
//===------------------------------------------------------------------------===

template <class Type_>
concept TrivialLayout = std::is_trivial_v<Type_> && std::is_standard_layout_v<Type_>;

// • is_trivial_layout
//
template <class Type_>
consteval bool is_trivial_layout(void) noexcept
{
    return false;
}

template <TrivialLayout Type_>
consteval bool is_trivial_layout(void) noexcept
{
    return true;
}

//===------------------------------------------------------------------------===
// • Alignment (always 16 bytes)
//===------------------------------------------------------------------------===

enum : uint32_t
{
    alignment = 16
};

constexpr bool is_aligned(uint32_t size_or_offset) noexcept
{
    return 0 == (size_or_offset & 0x0f);
}

template <typename Type_>
constexpr bool is_aligned(const Type_* memory) noexcept
{
    return 0 == (reinterpret_cast<uintptr_t>(memory) & 0x0f);
}

constexpr uint32_t aligned_size(uint32_t actual_size) noexcept
{
    return (actual_size + 0x0f) & ~0x0f;
}

template <TrivialLayout Type_>
constexpr uint32_t aligned_size(uint32_t capacity) noexcept
{
    return aligned_size( capacity*sizeof(Type_) );
}

//===------------------------------------------------------------------------===
// • Aligned concept
//===------------------------------------------------------------------------===

template <class Type_>
concept Aligned = ( 0 == (alignof(Type_) & 0x0f) );

template <typename Type_>
consteval bool is_aligned(void) noexcept
{
    return false;
}

template <Aligned Type_>
consteval bool is_aligned(void) noexcept
{
    return true;
}

template <typename Type_>
consteval uint32_t aligned_size(void) noexcept
{
    return static_cast<uint32_t>( (sizeof(Type_) + 0x0f) & ~0x0f  );
}

template <Aligned Type_>
consteval uint32_t aligned_size(void) noexcept
{
    return static_cast<uint32_t>( sizeof(Type_) );
}

template <TrivialLayout Type_>
constexpr uint32_t aligned_size(const Type_& ) noexcept
{
    return aligned_size<Type_>();
}

//===------------------------------------------------------------------------===
// Referential concept
//===------------------------------------------------------------------------===

template <class Ref_>
concept Referential = requires {

    // • Must be a struct or class
    std::is_class_v<Ref_>;

    // • value_type must be trivial layout
    data::is_trivial_layout<typename Ref_::value_type>();

    // • offset
    std::is_same_v<typeof(Ref_::offset), uint32_t>;
};

template <typename Type_>
constexpr bool is_referential(void) noexcept
{
    return false;
}

template <Referential Ref_>
constexpr bool is_referential(void) noexcept
{
    return true;
}

//===------------------------------------------------------------------------===
// • Concepts only available on host
//===------------------------------------------------------------------------===

#define TRIVIAL_LAYOUT data::TrivialLayout

//===------------------------------------------------------------------------===
// • Memory Layout Utilities (Host)
//===------------------------------------------------------------------------===

template <TrivialLayout Root_, TrivialLayout Type_>
uint32_t distance(const Root_* root, const Type_* data)
{
    return static_cast<uint32_t> (
                reinterpret_cast<const uint8_t*>(data) - reinterpret_cast<const uint8_t*>(root) );
}

template <TrivialLayout Type_, TrivialLayout Root_>
const Type_* offset_by(const Root_* root, uint32_t offset)
{
    return reinterpret_cast<const Type_*>(reinterpret_cast<const uint8_t*>(root) + offset);
}

template <TrivialLayout Type_, TrivialLayout Root_>
Type_* offset_by(Root_* root, uint32_t offset)
{
    return reinterpret_cast<Type_*>(reinterpret_cast<uint8_t*>(root) + offset);
}

#else // if defined ( __METAL_VERSION__ )

//===------------------------------------------------------------------------===
// • Concept names (Metal)
//===------------------------------------------------------------------------===

#define TRIVIAL_LAYOUT typename

//===------------------------------------------------------------------------===
// • Memory Layout Utilities (Metal)
//===------------------------------------------------------------------------===

template <TRIVIAL_LAYOUT Type_, TRIVIAL_LAYOUT Root_>
constant Type_* offset_by(constant Root_* root, uint32_t offset)
{
    return reinterpret_cast<constant Type_*>(reinterpret_cast<constant uint8_t*>(root) + offset);
}

template <TRIVIAL_LAYOUT Type_, TRIVIAL_LAYOUT Root_>
const device Type_* offset_by(const device Root_* root, uint32_t offset)
{
    return reinterpret_cast<const device Type_*>(reinterpret_cast<const device uint8_t*>(root) + offset);
}

template <TRIVIAL_LAYOUT Type_, TRIVIAL_LAYOUT Root_>
device Type_* offset_by(device Root_* root, uint32_t offset)
{
    return reinterpret_cast<device Type_*>(reinterpret_cast<device uint8_t*>(root) + offset);
}

#endif

} // namespace data
