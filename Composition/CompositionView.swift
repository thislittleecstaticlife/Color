//
//  CompositionView.swift
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

//===------------------------------------------------------------------------===
//
// MARK: - CompositionView
//
//===------------------------------------------------------------------------===

class CompositionView : ContentView {

    //===--------------------------------------------------------------------===
    // MARK: • Properties (Private)
    //
    private var isEditing = false

    //===--------------------------------------------------------------------===
    // MARK: • NSView Methods (Mouse/Trackpad)
    //
    override func scrollWheel(with event: NSEvent) {

        switch event.phase {
        case .began:

            beginEditing(with: event)

        case .changed:

            dragHueDial(by: event.scrollingDeltaX, with: event, scale: 0.5)

        case .ended, .cancelled:

            endEditing()

        default:
            break
        }
    }

    override func mouseDown(with event: NSEvent) {

        beginEditing(with: event)
    }

    override func mouseDragged(with event: NSEvent) {

        dragHueDial(by: event.deltaX, with: event, scale: 1.0)
    }

    override func mouseUp(with event: NSEvent) {

        endEditing()
    }

    //===--------------------------------------------------------------------===
    // MARK: • Private Methods (Mouse/Trackpad interaction)
    //
    private func beginEditing(with event: NSEvent) {

        let locationInView = convert(event.locationInWindow, from: nil)
        let hueDialFrame   = renderer.composition.hueDialFrame(in: self.bounds.size)

        if hueDialFrame.contains(locationInView) {

            isEditing = true
        }
    }

    private func endEditing() {

        if isEditing {

            isEditing = false
        }
    }

    private func dragHueDial(by deltaXInWindow: CGFloat, with event: NSEvent, scale: CGFloat) {

        guard isEditing else {
            return
        }

        let previousLocationInWindow = CGPoint( x: event.locationInWindow.x - deltaXInWindow,
                                                y: event.locationInWindow.y )

        let previousLocationInView = convert(previousLocationInWindow, from: nil)
        let locationInView         = convert(event.locationInWindow, from: nil)

        let dxScale      = CGFloat(event.modifierFlags.contains(.option) ? 1.0/16.0 : 1.0) * scale
        let dx           = (locationInView.x - previousLocationInView.x) * dxScale
        let hueDialFrame = renderer.composition.hueDialFrame(in: self.bounds.size)
        let hueDelta     = Float(-360.0 * dx / hueDialFrame.width)

        if 0.0 != hueDelta {

            let raw_hue = fmod(renderer.composition.hue + hueDelta, 360.0)

            renderer.composition.hue = (0.0 < raw_hue) ? raw_hue : raw_hue + 360.0

            needsDisplay = true
        }
    }
}
