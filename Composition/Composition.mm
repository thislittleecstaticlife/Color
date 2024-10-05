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

        *composition = {
            .grid_size = { 30, 30 },
            .jc_region = { .left = 1, .top = 1, .right = 29, .bottom = 29 },
            .hue = 42.794290425520614f, // 01 Red
//            .hue = 102.52116703710462f, // 02 Yellow
//            .hue = 136.26636667129654f, // 03 Green
//            .hue = 201.83718573465393f, // 04 Cyan
//            .hue = 258.64953857226578f, // 05 Blue
//            .hue = 325.26554587953854f, // 06 Magenta
        };
    }

    return self;
}

//===------------------------------------------------------------------------===
#pragma mark - Properties
//===------------------------------------------------------------------------===

- (void)setHue:(float)newHue {

    const auto reducedHue    = fmodf(newHue, 360.0f);
    const auto normalizedHue = (reducedHue < 0.0f) ? reducedHue + 360.0f : reducedHue;
    const auto composition   = [self currentComposition];

    if (composition->hue != normalizedHue) {

        auto updateComposition = [self nextComposition];

        updateComposition->hue = normalizedHue;
    }
}

- (float)hue {

    return [self currentComposition]->hue;
}

//===------------------------------------------------------------------------===
#pragma mark - Properties (Private)
//===------------------------------------------------------------------------===

- (CompositionData*)currentComposition {

    return static_cast<CompositionData*>(compositionBuffers[compositionBufferIndex].contents);
}

- (CompositionData*)nextComposition {

    compositionBufferIndex = (compositionBufferIndex + 1) % compositionBuffers.count;

    return [self currentComposition];
}

//===------------------------------------------------------------------------===
#pragma mark - Properties
//===------------------------------------------------------------------------===

- (nonnull id<MTLBuffer>)compositionBuffer {

    return compositionBuffers[compositionBufferIndex];
}

- (NSInteger)hueOffset {

    return offsetof(CompositionData, hue);
}

@end
