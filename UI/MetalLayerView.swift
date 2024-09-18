//
//  MetalLayerView.swift
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
import QuartzCore
import simd

//===------------------------------------------------------------------------===
//
// MARK: - MetalLayerView
//
//===------------------------------------------------------------------------===

class MetalLayerView : NSView {

    //===--------------------------------------------------------------------===
    // MARK: • Properties
    //
    let metalLayer = CAMetalLayer()

    var drawableSize : SIMD2<UInt32> {

        .init( UInt32(metalLayer.drawableSize.width),
               UInt32(metalLayer.drawableSize.height) )
    }

    //===--------------------------------------------------------------------===
    // MARK: • Initialization
    //
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {

        // • Layer-backed view
        //
        self.wantsLayer                = true
        self.layerContentsRedrawPolicy = .duringViewResize
    }

    //===--------------------------------------------------------------------===
    // MARK: • NSView Methods
    //
    override func makeBackingLayer() -> CALayer {

        return metalLayer
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        resizeDrawable()
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        resizeDrawable()
    }

    override func setBoundsSize(_ newSize: NSSize) {
        super.setBoundsSize(newSize)
        resizeDrawable()
    }

    override func viewWillMove(toWindow newWindow: NSWindow?) {
        super.viewWillMove(toWindow: newWindow)

        if let window, window !== newWindow {

            endObserving(window: window)
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()

        if let window {
            beginObserving(window: window)
        }
    }

    //===--------------------------------------------------------------------===
    // MARK: • Window notifications
    //
    func beginObserving(window: NSWindow) {

        let notificationCenter = NotificationCenter.default

        // • Full screen notifications
        //
        notificationCenter.addObserver( self, selector: #selector(didEnterFullScreen(_:)),
                                        name: NSWindow.didEnterFullScreenNotification,
                                        object: window )

        notificationCenter.addObserver( self, selector: #selector(willExitFullScreen(_:)),
                                        name: NSWindow.willExitFullScreenNotification,
                                        object:window )

        // • Window will close notification
        //
        notificationCenter.addObserver( self, selector: #selector(windowWillClose(_:)),
                                        name: NSWindow.willCloseNotification,
                                        object: window )
    }

    func endObserving(window: NSWindow) {

        let notificationCenter = NotificationCenter.default

        // • Full screen notifications
        //
        notificationCenter.removeObserver(self, name: NSWindow.willExitFullScreenNotification, object: window)
        notificationCenter.removeObserver(self, name: NSWindow.didEnterFullScreenNotification, object: window)

        // • Window will close notification
        //
        notificationCenter.removeObserver(self, name: NSWindow.willCloseNotification, object: window)
    }

    //===--------------------------------------------------------------------===
    // MARK: • Resizing
    //
    private func resizeDrawable() {

        guard let scale = self.window?.backingScaleFactor else {
            return
        }

        let newSize = CGSize(width: bounds.size.width * scale, height: bounds.size.height * scale)

        guard 0.0 < newSize.width && 0.0 < newSize.height && newSize != metalLayer.drawableSize else {
            return
        }

        metalLayer.drawableSize = newSize
        didResizeDrawable(newSize)
    }

    //===--------------------------------------------------------------------===
    // MARK: • Changes and Notifications
    //
    func didResizeDrawable(_ newSize: CGSize) {

        // • Nothing in base implementation
    }

    @objc func didEnterFullScreen(_ sender: Any?) {

        // • Nothing in base implementation
    }

    @objc func willExitFullScreen(_ sender: Any?) {

        // • Nothing in base implementation
    }

    @objc func windowWillClose(_ sender: Any?) {

        // • Nothing in base implementation
    }
}
