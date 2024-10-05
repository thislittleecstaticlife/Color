//
//  AppDelegate.swift
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
import UniformTypeIdentifiers

//===------------------------------------------------------------------------===
//
// MARK: - AppDelegate
//
//===------------------------------------------------------------------------===

class AppDelegate : NSObject, NSApplicationDelegate {

    //===--------------------------------------------------------------------===
    // MARK: • Properties (Private)
    //
    private var window      : NSWindow!
    private var renderer    : Renderer!
    private var contentView : ContentView!

    //===--------------------------------------------------------------------===
    // MARK: • NSApplicationDelegate Methods
    //
    func applicationDidFinishLaunching(_ aNotification: Notification) {

        // • Add a Quit menu item
        //
        guard let appName = Bundle.main.infoDictionary?[kCFBundleNameKey as String] as? String,
              let mainMenu = NSApp.mainMenu,
              let appMenuSubmenu = mainMenu.items.first?.submenu else {

            fatalError()
        }

        appMenuSubmenu.addItem( withTitle: "Quit " + appName,
                                action: #selector(NSApplication.terminate(_:)),
                                keyEquivalent: "q" )

        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem( withTitle: "Close Window", action:#selector(closeWindow), keyEquivalent:"w" )
        fileMenu.addItem( .separator() )
        fileMenu.addItem(withTitle: "Export...", action: #selector(exportImage), keyEquivalent: "e")

        let fileMenuItem = NSMenuItem()
        fileMenuItem.submenu = fileMenu

        mainMenu.addItem(fileMenuItem)

        // • Create the window
        //
        window = NSWindow( contentRect: .init(x: 0, y: 0, width: 960, height: 540),
                           styleMask: [.titled, .closable, .resizable, .miniaturizable],
                           backing: .buffered,
                           defer: false )

        // • Go ahead and set the title, but hide it
        //
        window.title           = appName
        window.titleVisibility = .hidden

        // • Frame auto-save name
        //
        window.setFrameAutosaveName(appName + ".Window")

        // • Metal resources and renderer
        //
        let maximumDrawableCount = 2

        guard let device = MTLCreateSystemDefaultDevice(),
              let library = device.makeDefaultLibrary(),
              let composition = Composition(device: device, bufferCount: maximumDrawableCount),
              let commandQueue = device.makeCommandQueue(),
              let renderer = Renderer(library: library, composition: composition,
                                      commandQueue: commandQueue) else {

            fatalError()
        }

        self.renderer = renderer

        // • Composition view
        //
        contentView = CompositionView( frame: .zero, renderer: renderer,
                                       commandQueue: commandQueue,
                                       maximumDrawableCount: maximumDrawableCount )

        window.contentView = contentView

        NSLayoutConstraint.activate([
            contentView.widthAnchor.constraint(equalToConstant: 540),
            contentView.heightAnchor.constraint(equalToConstant: 598),
        ])

        window.makeKeyAndOrderFront(nil)
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {

        return true
    }

    //===--------------------------------------------------------------------===
    // MARK: • Actions (Private)
    //
    @objc private func closeWindow() {

        window?.close()
    }

    @objc private func exportImage() {

        // • Redraw the current frame
        //
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2D
        textureDescriptor.pixelFormat = renderer.pixelFormat
        textureDescriptor.width       = 1080
        textureDescriptor.height      = 1188
        textureDescriptor.usage       = .renderTarget

        guard let texture = renderer.device.makeTexture(descriptor: textureDescriptor),
              let commandQueue = renderer.device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {

                return
        }

        let didDraw = renderer.draw(to: texture, with: commandBuffer)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        guard didDraw else {
            return
        }

        // • Create an NSImage from the texture
        //
        guard let bitmapDescription = BitmapDescription(from: texture, colorspace: renderer.colorspace),
              let bitmap = bitmapDescription.makeContext(),
              let bitmapData = bitmap.data else {

            return
        }

        texture.getBytes(bitmapData,
                         bytesPerRow: bitmap.bytesPerRow,
                         from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                         mipmapLevel: 0)

        // • Export the image as PNG
        //
        guard let image = bitmap.makeImage(),
              let picturesDirectoryURL = try? FileManager.default.url(for: .picturesDirectory,
                                                                      in: .userDomainMask,
                                                                      appropriateFor: nil,
                                                                      create: false) else {
            return
        }

        DispatchQueue.global().async {

            let formatter = DateFormatter()

            formatter.locale     = .init(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd' at 'h.mm.ss a"

            let date      = formatter.string(from: Date.now)
            let filename  = "Color Still " + date + ".png"
            let outputURL = picturesDirectoryURL.appending(component: filename)

            guard let destination = CGImageDestinationCreateWithURL(outputURL as CFURL,
                                                                    UTType.png.identifier as CFString,
                                                                    1,
                                                                    nil) else {
                return
            }

            CGImageDestinationAddImage(destination, image, nil)
            CGImageDestinationFinalize(destination)
        }
    }
}
