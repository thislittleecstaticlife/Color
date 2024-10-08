//
//  Renderer.swift
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

import Foundation
import Metal
import QuartzCore

//===------------------------------------------------------------------------===
//
// MARK: - Renderer
//
//===------------------------------------------------------------------------===

class Renderer {

    //===--------------------------------------------------------------------===
    // MARK: • Properties (Read-Only)
    //
    let pixelFormat = MTLPixelFormat.rgba16Float
    let depthFormat = MTLPixelFormat.depth32Float
    let colorspace  : CGColorSpace
    let device      : MTLDevice
    let composition : Composition

    //===--------------------------------------------------------------------===
    // MARK: • Properties (Private)
    //
    private let generateVerticesPipelineState : MTLComputePipelineState
    private let vertexBuffer                  : MTLBuffer
    private let foregroundPipelineState       : MTLRenderPipelineState
    private let backgroundPipelineState       : MTLRenderPipelineState
    private let maxCPipelineState             : MTLRenderPipelineState
    private let hueGradientPipelineState      : MTLRenderPipelineState
    private let hueGradientTexture            : MTLTexture
    private let depthState                    : MTLDepthStencilState
    private var textures                      : (multisample: MTLTexture, depth: MTLTexture)?

    //===--------------------------------------------------------------------===
    // MARK: • Constants (Private)
    //
    private let vertexCount = 129

    //===--------------------------------------------------------------------===
    // MARK: • Initilization
    //
    init?(library: MTLLibrary, composition: Composition, commandQueue: MTLCommandQueue) {

        self.device      = library.device
        self.composition = composition

        // • Color space
        //
        guard let colorspace = CGColorSpace(name: CGColorSpace.linearDisplayP3) else {
            return nil
        }

        // • Generate vertices pipeline
        //
        guard let generateVertices = library.makeFunction(name: "generate_vertices"),
              let generateVerticesPipelineState =
                try? device.makeComputePipelineState(function: generateVertices) else {

            return nil
        }

        // • Vertex buffer
        //
        let vertexBufferLength = MemoryLayout<SIMD4<Float>>.stride * vertexCount

        guard let vertexBuffer = device.makeBuffer(length: vertexBufferLength) else {
            return nil
        }

        // • Foreground pipeline
        //
        guard let foregroundVertex = library.makeFunction(name: "foreground_vertex"),
              let foregroundFragment = library.makeFunction(name: "foreground_fragment") else {

            return nil
        }

        let foregroundPipelineDescriptor = MTLRenderPipelineDescriptor()
        foregroundPipelineDescriptor.colorAttachments[0].pixelFormat = self.pixelFormat
        foregroundPipelineDescriptor.depthAttachmentPixelFormat      = self.depthFormat
        foregroundPipelineDescriptor.vertexFunction                  = foregroundVertex
        foregroundPipelineDescriptor.fragmentFunction                = foregroundFragment
        foregroundPipelineDescriptor.rasterSampleCount               = 4

        guard let foregroundPipelineState =
                try? device.makeRenderPipelineState(descriptor: foregroundPipelineDescriptor) else {

            return nil
        }

        // • Background pipeline
        //
        guard let backgroundVertex = library.makeFunction(name: "background_vertex"),
              let backgroundFragment = library.makeFunction(name: "background_fragment") else {

            return nil
        }

        let backgroundPipelineDescriptor = MTLRenderPipelineDescriptor()
        backgroundPipelineDescriptor.colorAttachments[0].pixelFormat = self.pixelFormat
        backgroundPipelineDescriptor.depthAttachmentPixelFormat      = self.depthFormat
        backgroundPipelineDescriptor.vertexFunction                  = backgroundVertex
        backgroundPipelineDescriptor.fragmentFunction                = backgroundFragment
        backgroundPipelineDescriptor.rasterSampleCount               = 4

        guard let backgroundPipelineState =
                try? device.makeRenderPipelineState(descriptor: backgroundPipelineDescriptor) else {

            return nil
        }

        // • Max chroma color pipeline
        //
        guard let maxCVertex = library.makeFunction(name: "max_c_vertex"),
              let passThroughFragment = library.makeFunction(name: "pass_through_fragment") else {

            return nil
        }

        let maxCPipelineDescriptor = MTLRenderPipelineDescriptor()
        maxCPipelineDescriptor.colorAttachments[0].pixelFormat = self.pixelFormat
        maxCPipelineDescriptor.depthAttachmentPixelFormat      = self.depthFormat
        maxCPipelineDescriptor.vertexFunction                  = maxCVertex
        maxCPipelineDescriptor.fragmentFunction                = passThroughFragment
        maxCPipelineDescriptor.rasterSampleCount               = 4

        guard let maxCPipelineState =
                try? device.makeRenderPipelineState(descriptor: maxCPipelineDescriptor) else {

            return nil
        }

        // • Hue gradient pipeline
        //
        guard let hueGradientVertex = library.makeFunction(name: "hue_gradient_vertex"),
              let hueGradientFragment = library.makeFunction(name: "hue_gradient_fragment") else {

            return nil
        }

        let hueGradientPipelineDescriptor = MTLRenderPipelineDescriptor()
        hueGradientPipelineDescriptor.colorAttachments[0].pixelFormat = self.pixelFormat
        hueGradientPipelineDescriptor.depthAttachmentPixelFormat      = self.depthFormat
        hueGradientPipelineDescriptor.vertexFunction                  = hueGradientVertex
        hueGradientPipelineDescriptor.fragmentFunction                = hueGradientFragment
        hueGradientPipelineDescriptor.rasterSampleCount               = 4

        guard let hueGradientPipelineState =
                try? device.makeRenderPipelineState(descriptor: hueGradientPipelineDescriptor) else {

            return nil
        }

        // • Hue gradient texture
        //
        let hueGradientTextureDescriptor = MTLTextureDescriptor()
        hueGradientTextureDescriptor.pixelFormat = self.pixelFormat
        hueGradientTextureDescriptor.textureType = .type1D
        hueGradientTextureDescriptor.width       = 1024
        hueGradientTextureDescriptor.height      = 1
        hueGradientTextureDescriptor.sampleCount = 1
        hueGradientTextureDescriptor.usage       = [.shaderRead, .shaderWrite]

        guard let hueGradientTexture =
                device.makeTexture(descriptor: hueGradientTextureDescriptor) else {

            return nil
        }

        // • Depth/stencil state
        //
        let depthStencilDescriptor = MTLDepthStencilDescriptor()
        depthStencilDescriptor.depthCompareFunction = .less
        depthStencilDescriptor.isDepthWriteEnabled  = true

        guard let depthState =
                device.makeDepthStencilState(descriptor: depthStencilDescriptor) else {

            return nil
        }

        // • Assign properties
        //
        self.colorspace                    = colorspace
        self.generateVerticesPipelineState = generateVerticesPipelineState
        self.vertexBuffer                  = vertexBuffer
        self.foregroundPipelineState       = foregroundPipelineState
        self.backgroundPipelineState       = backgroundPipelineState
        self.maxCPipelineState             = maxCPipelineState
        self.hueGradientPipelineState      = hueGradientPipelineState
        self.hueGradientTexture            = hueGradientTexture
        self.depthState                    = depthState

        // • Create one-time GPU-rendered resources
        //
        guard initializeResources(with: library, commandQueue: commandQueue) else {
            return nil
        }
    }

    //===--------------------------------------------------------------------===
    // MARK: • Methods
    //
    @discardableResult
    func draw(to outputTexture: MTLTexture, with commandBuffer: MTLCommandBuffer) -> Bool {

        // • Memoryless multi-sample and depth textures
        //
        guard let (multisampleTexture, depthTexture) = intermediateTextures(for: outputTexture) else {
            return false
        }

        // • Current composition buffer
        //
        let compositionBuffer = composition.prepareBuffer()

        // • Generate Jzazbz volume slice vertices at the current hue
        //
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }

        computeEncoder.setComputePipelineState(generateVerticesPipelineState)
        computeEncoder.setBuffer(compositionBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 1)

        let threadsWidth  = generateVerticesPipelineState.threadExecutionWidth
        let threadsHeight = generateVerticesPipelineState.maxTotalThreadsPerThreadgroup / threadsWidth

        let threadsPerThreadgroup = MTLSize(width: threadsWidth, height: threadsHeight, depth: 1)

        computeEncoder.dispatchThreads( .init(width:  threadsWidth,
                                              height: vertexCount / 2,
                                              depth:  1),
                                        threadsPerThreadgroup: threadsPerThreadgroup )
        computeEncoder.endEncoding()

        // • Render pass
        //
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture        = multisampleTexture
        renderPassDescriptor.colorAttachments[0].resolveTexture = outputTexture
        renderPassDescriptor.colorAttachments[0].clearColor     = MTLClearColorMake(0, 0, 0, 1)
        renderPassDescriptor.colorAttachments[0].loadAction     = .clear
        renderPassDescriptor.colorAttachments[0].storeAction    = .multisampleResolve

        renderPassDescriptor.depthAttachment.texture     = depthTexture
        renderPassDescriptor.depthAttachment.clearDepth  = 1.0
        renderPassDescriptor.depthAttachment.loadAction  = .clear
        renderPassDescriptor.depthAttachment.storeAction = .dontCare

        guard let renderEncoder =
                commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {

            return false
        }

        renderEncoder.setDepthStencilState(depthState)
        renderEncoder.setVertexBuffer(compositionBuffer, offset: 0, index: 0)

        // • Render the foreground
        //
        renderEncoder.setRenderPipelineState(foregroundPipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: vertexCount)

        // -  then the background
        //
        renderEncoder.setRenderPipelineState(backgroundPipelineState)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // • Hue gradient
        //
        renderEncoder.setRenderPipelineState(hueGradientPipelineState)
        renderEncoder.setFragmentTexture(hueGradientTexture, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // • Max chroma color indicator
        //
        renderEncoder.setRenderPipelineState(maxCPipelineState)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // • Done
        //
        renderEncoder.endEncoding()

        return true
    }

    //===--------------------------------------------------------------------===
    // MARK: • Private Methods
    //
    private func initializeResources(with library: MTLLibrary, commandQueue: MTLCommandQueue) -> Bool {

        // • Generate hue gradient texture
        //
        guard let generateHueGradient = library.makeFunction(name: "generate_hue_gradient"),
              let generateHueGradientPipeline =
                try? device.makeComputePipelineState(function: generateHueGradient),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {

            return false
        }

        computeEncoder.setComputePipelineState(generateHueGradientPipeline)
        computeEncoder.setTexture(hueGradientTexture, index: 0)

        let threadsWidth  = generateHueGradientPipeline.threadExecutionWidth
        let threadsHeight = generateHueGradientPipeline.maxTotalThreadsPerThreadgroup / threadsWidth

        let threadsPerThreadGroup = MTLSize(width: threadsWidth, height: threadsHeight, depth: 1)
        let threads = MTLSize(width: threadsWidth, height: hueGradientTexture.width, depth: 1)

        computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadGroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()

        return true
    }

    private func intermediateTextures(for outputTexture: MTLTexture) -> (MTLTexture, MTLTexture)? {

        if let textures,
           textures.multisample.width == outputTexture.width,
           textures.multisample.height == outputTexture.height {

            return textures
        }

        // • Multi-sample texture descriptor
        //
        let multisampleTextureDescriptor = MTLTextureDescriptor()
        multisampleTextureDescriptor.textureType = .type2DMultisample
        multisampleTextureDescriptor.pixelFormat = self.pixelFormat
        multisampleTextureDescriptor.usage       = .renderTarget
        multisampleTextureDescriptor.storageMode = .memoryless
        multisampleTextureDescriptor.width       = outputTexture.width
        multisampleTextureDescriptor.height      = outputTexture.height
        multisampleTextureDescriptor.sampleCount = 4

        // • Depth texture descriptor (only pixel format differs from multisample texture)
        //
        let depthTextureDecriptor = multisampleTextureDescriptor.copy() as! MTLTextureDescriptor
        depthTextureDecriptor.pixelFormat = self.depthFormat

        guard let multisampleTexture = device.makeTexture(descriptor: multisampleTextureDescriptor),
            let depthTexture = device.makeTexture(descriptor: depthTextureDecriptor) else {

            return nil
        }

        textures = (multisampleTexture, depthTexture)

        return textures
    }
}
