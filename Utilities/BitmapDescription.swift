//
//  BitmapDescription.swift
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

import QuartzCore
import Metal

//===--------------------------------------------------------------------===
// MARK: • AlphaType
//
enum AlphaType {

    case premultiplied
    case normal
    case none

    fileprivate func alphaInfo(none: CGImageAlphaInfo, normal: CGImageAlphaInfo, premultiplied: CGImageAlphaInfo)
    -> CGImageAlphaInfo {

        switch self {
        case .premultiplied: return premultiplied
        case .normal:        return normal
        case .none:          return none
        }
    }

    fileprivate var firstAlphaInfo : CGImageAlphaInfo {

        return alphaInfo(none: .noneSkipFirst, normal: .first, premultiplied: .premultipliedFirst)
    }

    fileprivate var lastAlphaInfo : CGImageAlphaInfo {

        return alphaInfo(none: .noneSkipLast, normal: .last, premultiplied: .premultipliedLast)
    }
}

//===------------------------------------------------------------------------===
// MARK: - BitmapPixelDescription
//===------------------------------------------------------------------------===

struct BitmapPixelDescription {

    //===--------------------------------------------------------------------===
    // MARK: • Properties
    //
    let bitsPerComponent : Int
    let bytesPerPixel    : Int
    let bitmapInfo       : CGBitmapInfo
    let alphaInfo        : CGImageAlphaInfo

    //===--------------------------------------------------------------------===
    // MARK: • Initialization (Private)
    //
    private init(bitsPerComponent: Int, bytesPerPixel: Int, bitmapInfo: CGBitmapInfo, alphaInfo: CGImageAlphaInfo) {

        self.bitsPerComponent = bitsPerComponent
        self.bytesPerPixel    = bytesPerPixel
        self.bitmapInfo       = bitmapInfo
        self.alphaInfo        = alphaInfo
    }

    //===--------------------------------------------------------------------===
    // MARK: • Initialization
    //
    init?(pixelFormat: MTLPixelFormat, alphaType: AlphaType = .none) {

        switch pixelFormat {

        case .a8Unorm, .r8Sint, .r8Snorm, .r8Uint, .r8Unorm:

            precondition( .none == alphaType )

            self.init(bitsPerComponent: 8, bytesPerPixel: 1, bitmapInfo: [], alphaInfo: .none)

        case .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint:

            precondition( .none == alphaType )

            self.init(bitsPerComponent: 8, bytesPerPixel: 2, bitmapInfo: [], alphaInfo: .none)

        case .bgra8Unorm:

            self.init( bitsPerComponent: 8, bytesPerPixel: 4, bitmapInfo: [.byteOrder32Little],
                       alphaInfo: alphaType.firstAlphaInfo )

        case .rgba16Float:

            self.init( bitsPerComponent: 16, bytesPerPixel: 8,
                       bitmapInfo: [.floatComponents, .byteOrder16Little],
                       alphaInfo: alphaType.lastAlphaInfo )

        default:
            //  - none others currently supported
            return nil
        }
    }
}

//===------------------------------------------------------------------------===
// MARK: - BitmapPixelDescription : Equatable
//===------------------------------------------------------------------------===

extension BitmapPixelDescription : Equatable {

    static func == (lhs: BitmapPixelDescription, rhs: BitmapPixelDescription) -> Bool {

        return lhs.bitsPerComponent == rhs.bitsPerComponent
            && lhs.bytesPerPixel    == rhs.bytesPerPixel
            && lhs.bitmapInfo       == rhs.bitmapInfo
            && lhs.alphaInfo        == rhs.alphaInfo
    }
}

//===------------------------------------------------------------------------===
// MARK: - extension BitmapPixelDescription (Pixel Types)
//===------------------------------------------------------------------------===

extension BitmapPixelDescription {

    //===--------------------------------------------------------------------===
    // MARK: • Methods
    //
    func matches(_ context: CGContext) -> Bool {

        return self.bytesPerPixel*8  == context.bitsPerPixel
            && self.bitsPerComponent == context.bitsPerComponent
            && self.bitmapInfo       == context.bitmapInfo
            && self.alphaInfo        == context.alphaInfo
    }

    //===--------------------------------------------------------------------===
    // MARK: • Static Methods (Pixel Types)
    //
    static var r8 : BitmapPixelDescription {

        return .init(bitsPerComponent: 8, bytesPerPixel: 1, bitmapInfo: [], alphaInfo: .none)
    }

    static var rg8 : BitmapPixelDescription {

        return .init(bitsPerComponent: 8, bytesPerPixel: 2, bitmapInfo: [], alphaInfo: .none)
    }

    static func bgra8(alpha: AlphaType) -> BitmapPixelDescription {

        return .init( bitsPerComponent: 8, bytesPerPixel: 4,
                      bitmapInfo: [.byteOrder32Little],
                      alphaInfo : alpha.firstAlphaInfo )
    }

    static func rgba16(alpha: AlphaType) -> BitmapPixelDescription {

        return .init( bitsPerComponent: 16, bytesPerPixel: 8,
                      bitmapInfo: [.byteOrder16Little],
                      alphaInfo : alpha.lastAlphaInfo )
    }

    static func rgba16Float(alpha: AlphaType) -> BitmapPixelDescription {

        return .init( bitsPerComponent: 16, bytesPerPixel: 8,
                      bitmapInfo: [.floatComponents, .byteOrder16Little],
                      alphaInfo : alpha.lastAlphaInfo )
    }

    static func rgba32Float(alpha: AlphaType) -> BitmapPixelDescription {

        return .init( bitsPerComponent: 32, bytesPerPixel: 16,
                      bitmapInfo: [.floatComponents, .byteOrder32Little],
                      alphaInfo : alpha.lastAlphaInfo )
    }

    //===--------------------------------------------------------------------===
    // MARK: • Static Methods (Pixel Types - No Alpha)
    //
    static var bgrx8       = BitmapPixelDescription.bgra8(alpha: .none)
    static var rgbx16      = BitmapPixelDescription.rgba16(alpha: .none)
    static var rgbx16Float = BitmapPixelDescription.rgba16Float(alpha: .none)
    static var rgbx32Float = BitmapPixelDescription.rgba32Float(alpha: .none)
}

//===------------------------------------------------------------------------===
// MARK: - BitmapDescription
//===------------------------------------------------------------------------===

struct BitmapDescription {

    //===--------------------------------------------------------------------===
    // MARK: • Properties
    //
    let width            : Int
    let height           : Int

    let bytesPerRow      : Int
    let bitsPerComponent : Int
    let bytesPerPixel    : Int
    let bitmapInfo       : CGBitmapInfo
    let alphaInfo        : CGImageAlphaInfo
    let colorspace       : CGColorSpace

    //===--------------------------------------------------------------------===
    // MARK: • Initialization
    //
    init?(width: Int, height: Int, pixel: BitmapPixelDescription, colorspace: CGColorSpace) {

        guard 1 <= width && 1 <= height else  {
            return nil
        }

        self.width            = width
        self.height           = height
        self.bytesPerRow      = ((width * pixel.bytesPerPixel) + 63) & ~63
        self.bitsPerComponent = pixel.bitsPerComponent
        self.bytesPerPixel    = pixel.bytesPerPixel
        self.bitmapInfo       = pixel.bitmapInfo
        self.alphaInfo        = pixel.alphaInfo
        self.colorspace       = colorspace
    }

    init?(width: Int, height: Int, pixel: BitmapPixelDescription, colorSpaceName: CFString) {

        guard let colorspace = CGColorSpace(name: colorSpaceName) else {
            return nil
        }

        self.init(width: width, height: height, pixel: pixel, colorspace: colorspace)
    }

    init?(size: CGSize, pixel: BitmapPixelDescription, colorspace: CGColorSpace) {

        self.init( width: Int(size.width), height: Int(size.height), pixel: pixel,
                   colorspace: colorspace )
    }

    init?(from texture: MTLTexture, colorspace: CGColorSpace, alphaType: AlphaType = .none) {

        guard let pixel = BitmapPixelDescription(pixelFormat: texture.pixelFormat,
                                                 alphaType: alphaType) else {
            return nil
        }

        self.init( width: texture.width, height: texture.height, pixel: pixel,
                   colorspace: colorspace )
    }
}

//===------------------------------------------------------------------------===
// MARK: - BitmapDescription : Equatable
//===------------------------------------------------------------------------===

extension BitmapDescription : Equatable {

    static func == (lhs: BitmapDescription, rhs: BitmapDescription) -> Bool {

        return lhs.width            == rhs.width
            && lhs.height           == rhs.height
            && lhs.bitsPerComponent == rhs.bitsPerComponent
            && lhs.bytesPerPixel    == rhs.bytesPerPixel
            && lhs.bitmapInfo       == rhs.bitmapInfo
            && lhs.alphaInfo        == rhs.alphaInfo
            && CFEqual(lhs.colorspace, rhs.colorspace)
    }
}

//===------------------------------------------------------------------------===
// MARK: - extension BitmapDescription (Utilities)
//===------------------------------------------------------------------------===

extension BitmapDescription {

    //===--------------------------------------------------------------------===
    // MARK: • Properties
    //
    var bufferSize : Int {

        return bytesPerRow * height
    }

    var combinedBitmapInfo : UInt32 {

        return bitmapInfo.rawValue | alphaInfo.rawValue
    }

    //===--------------------------------------------------------------------===
    // MARK: • Methods
    //
    func makeContext() -> CGContext? {

        return CGContext( data: nil, width: self.width, height: self.height,
                          bitsPerComponent: self.bitsPerComponent, bytesPerRow: self.bytesPerRow,
                          space: self.colorspace, bitmapInfo: self.combinedBitmapInfo )
    }
}
