//
//  Composition.m
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

#import "Composition.h"
#import "CompositionData.hpp"

#import <Graphics/Jzazbz.hpp>

#import <numeric>

//===------------------------------------------------------------------------===
//
#pragma mark - Composition Implementation
//
//===------------------------------------------------------------------------===

@implementation Composition
{
    NSArray<id<MTLBuffer>> *compositionBuffers;
    NSInteger               compositionBufferIndex;
    float                   nextHue;
}

//===------------------------------------------------------------------------===
#pragma mark - Initialization
//===------------------------------------------------------------------------===

- (nullable instancetype)initWithDevice:(nonnull id<MTLDevice>)device
                            bufferCount:(NSInteger)bufferCount {

    self = [super init];

    if (nil != self) {

        // • Composition buffer
        //
        const auto compositionBufferLength = data::aligned_size<CompositionData>();

        auto compositionBuffers = [[NSMutableArray<id<MTLBuffer>> alloc] initWithCapacity:bufferCount];

        for (NSInteger ib = 0; ib < bufferCount; ++ib) {

            auto compositionBuffer = [device newBufferWithLength:compositionBufferLength
                                                         options:0];
            if (nil == compositionBuffer) {
                return nil;
            }

            [compositionBuffers addObject:compositionBuffer];
        }

        self->compositionBuffers = compositionBuffers;
        compositionBufferIndex   = 0;

        // • Initialize the first buffer and copy to the others
        //
        auto composition = [self currentComposition];

        nextHue = 42.794290425520614f; // 01 Red
        // nextHue = 102.52116703710462f; // 02 Yellow
        // nextHue = 136.26636667129654f; // 03 Green
        // nextHue = 201.83718573465393f; // 04 Cyan
        // nextHue = 258.64953857226578f; // 05 Blue
        // nextHue = 325.26554587953854f; // 06 Magenta

        *composition = {
            .grid_size       = { 30, 33 },
            .jc_region       = { .left =  1, .top =  1, .right = 29, .bottom = 29 },
            .gradient_region = { .left =  1, .top = 30, .right = 26, .bottom = 32 },
            .max_c_region    = { .left = 27, .top = 30, .right = 29, .bottom = 32 },
            .hue             = nextHue,
            .max_c_color     = jzazbz::find_max_chroma_color(nextHue)
        };

        for (NSInteger ib = 1; ib < compositionBuffers.count; ++ib) {

            memcpy( compositionBuffers[ib].contents, compositionBuffers[0].contents,
                   compositionBufferLength );
        }
    }

    return self;
}

//===------------------------------------------------------------------------===
#pragma mark - Properties
//===------------------------------------------------------------------------===

- (void)setHue:(float)newHue {

    const auto reducedHue = fmodf(newHue, 360.0f);

    nextHue = (reducedHue < 0.0f) ? reducedHue + 360.0f : reducedHue;
}

- (float)hue {

    return nextHue;
}

//===------------------------------------------------------------------------===
#pragma mark - Properties (Private)
//===------------------------------------------------------------------------===

- (nonnull CompositionData*)currentComposition {

    return static_cast<CompositionData*>(compositionBuffers[compositionBufferIndex].contents);
}

//===------------------------------------------------------------------------===
#pragma mark - Methods
//===------------------------------------------------------------------------===

- (nonnull id<MTLBuffer>)prepareCompositionBuffer {

    auto composition = [self currentComposition];

    if (composition->hue != nextHue) {

        compositionBufferIndex = (compositionBufferIndex + 1) % compositionBuffers.count;
        composition            = [self currentComposition];

        composition->hue         = nextHue;
        composition->max_c_color = jzazbz::find_max_chroma_color(nextHue);
    }

    return compositionBuffers[compositionBufferIndex];
}

- (CGRect)hueDialFrameInViewOfSize:(CGSize)viewSize {

    const auto composition = [self currentComposition];

    const auto left   = (composition->gradient_region.left   * viewSize.width)  / composition->grid_size.x;
    const auto right  = (composition->gradient_region.right  * viewSize.width)  / composition->grid_size.x;
    const auto top    = (composition->gradient_region.top    * viewSize.height) / composition->grid_size.y;
    const auto bottom = (composition->gradient_region.bottom * viewSize.height) / composition->grid_size.y;

    // • NSView bottom-up rect
    //
    return CGRectMake(left, viewSize.height - bottom, right - left, bottom - top);
}

@end
