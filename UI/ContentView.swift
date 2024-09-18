//
//  ContentView.swift
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

import Cocoa
import Metal

//===------------------------------------------------------------------------===
//
// MARK: - ContentView
//
//===------------------------------------------------------------------------===

class ContentView : MetalLayerView {

    //===--------------------------------------------------------------------===
    // MARK: • Properties (Private)
    //
    private let renderer     : Renderer
    private let commandQueue : MTLCommandQueue
    private var semaphore    : DispatchSemaphore

    //===--------------------------------------------------------------------===
    // MARK: • Initialization
    //
    @available(*, unavailable) override init(frame frameRect: NSRect) {
        fatalError()
    }

    @available(*, unavailable) required init?(coder: NSCoder) {
        fatalError()
    }

    init( frame: NSRect, renderer: Renderer, commandQueue: MTLCommandQueue,
          maximumDrawableCount: Int = 2 ) {

        precondition( 2 == maximumDrawableCount || 3 == maximumDrawableCount )

        self.renderer     = renderer
        self.commandQueue = commandQueue
        self.semaphore    = .init(value: maximumDrawableCount)

        super.init(frame: frame)

        // • Configure layer
        //
        metalLayer.device               = renderer.device
        metalLayer.colorspace           = renderer.colorspace
        metalLayer.pixelFormat          = renderer.pixelFormat
        metalLayer.maximumDrawableCount = maximumDrawableCount
        metalLayer.framebufferOnly      = true
        metalLayer.delegate             = self
    }

    //===--------------------------------------------------------------------===
    // MARK: • Changes and Notifications
    //
    override func didResizeDrawable(_ newSize: CGSize) {

        metalLayer.setNeedsDisplay()
    }
}

//===------------------------------------------------------------------------===
// MARK: - ContentView : CALayerDelegate
//===------------------------------------------------------------------------===

extension ContentView : CALayerDelegate {

    func display(_ layer: CALayer) {

        semaphore.wait()

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            semaphore.signal()
            return
        }

        if let drawable = metalLayer.nextDrawable() {

            renderer.draw(to: drawable.texture, with: commandBuffer)

            commandBuffer.present(drawable)
        }

        commandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        commandBuffer.commit()
    }
}
