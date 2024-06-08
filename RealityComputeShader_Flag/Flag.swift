/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Flag simulation Metal compute wrapper.
*/

// References to Metal do not compile for the Simulator.
//#if !targetEnvironment(simulator)
import Foundation
import Metal
import RealityKit

struct SimulationData {
    var wind: SIMD3<Float>
    var pad1: Float = 0
    
    init(wind: SIMD3<Float>) {
        self.wind = wind
    }
}

struct ClothData {
    var clothEntity: ModelEntity
    var meshData: ClothSimMetalNode
    
    init(clothEntity: ModelEntity, meshData: ClothSimMetalNode) {
        self.clothEntity = clothEntity
        self.meshData = meshData
    }
}

class ClothSimMetalNode {
    let vb1: MTLBuffer
    let vb2: MTLBuffer
    let normalBuffer: MTLBuffer
    let normalWorkBuffer: MTLBuffer
    let vertexCount: Int
    
    var velocityBuffers = [MTLBuffer]()
    
    var currentBufferIndex: Int = 0
    
    init(device: MTLDevice, width: uint, height: uint) {
        var vertices: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        var indices: [UInt32] = []
        
        for y in 0..<height {
            for x in 0..<width {
                let p = SIMD3<Float>(Float(x), 0, Float(y))
                vertices.append(p)
                normals.append(SIMD3<Float>(0, 1, 0))
                uvs.append(SIMD2<Float>(p.x / Float(width), p.z / Float(height)))
            }
        }
        
        for y in 0..<(height - 1) {
            for x in 0..<(width - 1) {
                // make 2 triangles from the 4 vertices of a quad
                let i0 = y * width + x
                let i1 = i0 + 1
                let i2 = i0 + width
                let i3 = i2 + 1
                
                // triangle 1
                indices.append(i0)
                indices.append(i2)
                indices.append(i3)
                
                // triangle 2
                indices.append(i0)
                indices.append(i3)
                indices.append(i1)
            }
        }
        
        let vertexBuffer1 = device.makeBuffer(bytes: vertices,
                                              length: vertices.count * MemoryLayout<SIMD3<Float>>.size,
                                              options: [.storageModeShared])
        let length = vertices.count
        let float4Ptr = vertexBuffer1?.contents().bindMemory(to: SIMD3<Float>.self,capacity: length)
        let float4Buffer = UnsafeBufferPointer(start: float4Ptr,count: length)
        let output = Array(float4Buffer)
        
        let vertexBuffer2 = device.makeBuffer(length: vertices.count * MemoryLayout<SIMD3<Float>>.size,
                                              options: [.cpuCacheModeWriteCombined])
        
//        let vertexSource = SCNGeometrySource(buffer: vertexBuffer1!,
//                                             vertexFormat: .float3,
//                                             semantic: .vertex,
//                                             vertexCount: vertices.count,
//                                             dataOffset: 0,
//                                             dataStride: MemoryLayout<float3>.size)
        
        let normalBuffer = device.makeBuffer(bytes: normals,
                                             length: normals.count * MemoryLayout<SIMD3<Float>>.size,
                                             options: [.storageModeShared])
        
        let normalWorkBuffer = device.makeBuffer(length: normals.count * MemoryLayout<SIMD3<Float>>.size,
                                                 options: [.cpuCacheModeWriteCombined])
        
//        let normalSource = SCNGeometrySource(buffer: normalBuffer!,
//                                             vertexFormat: .float3,
//                                             semantic: .normal,
//                                             vertexCount: normals.count,
//                                             dataOffset: 0,
//                                             dataStride: MemoryLayout<float3>.size)

//        let indexElement = SCNGeometryElement(indices: indices, primitiveType: .triangles)
//        let geo = SCNGeometry(sources: [vertexSource, normalSource, uvSource], elements: [indexElement])
        
        // velocity buffers
        let velocityBuffer1 = device.makeBuffer(length: vertices.count * MemoryLayout<SIMD3<Float>>.size,
                                                options: [.cpuCacheModeWriteCombined])
        
        let velocityBuffer2 = device.makeBuffer(length: vertices.count * MemoryLayout<SIMD3<Float>>.size,
                                                options: [.cpuCacheModeWriteCombined])
        
        self.vertexCount = vertices.count
        self.vb1 = vertexBuffer1!
        self.vb2 = vertexBuffer2!
        self.normalBuffer = normalBuffer!
        self.normalWorkBuffer = normalWorkBuffer!
        self.velocityBuffers = [velocityBuffer1!, velocityBuffer2!]
    }
    @MainActor
    func generateMeshResource() -> MeshResource? {
        do {
            return try MeshResource.generate(from: MeshResource.Contents())
        } catch  {
            print(error)
        }
        return nil
    }
}

/*
 Encapsulate the 'Metal stuff' within a single class to handle setup and execution
 of the compute shaders.
 */
class MetalClothSimulator {
    let device: MTLDevice
    
    let commandQueue: MTLCommandQueue
    let defaultLibrary: MTLLibrary
    let functionClothSim: MTLFunction
    let functionNormalUpdate: MTLFunction
    let functionNormalSmooth: MTLFunction
    let pipelineStateClothSim: MTLComputePipelineState
    let pipelineStateNormalUpdate: MTLComputePipelineState
    let pipelineStateNormalSmooth: MTLComputePipelineState

    let width: uint = 32
    let height: uint = 20
    
    var clothData = [ClothData]()
    
    init(device: MTLDevice) {
        self.device = device
        
        commandQueue = device.makeCommandQueue()!
        
        defaultLibrary = device.makeDefaultLibrary()!
        functionClothSim = defaultLibrary.makeFunction(name: "updateVertex")!
        functionNormalUpdate = defaultLibrary.makeFunction(name: "updateNormal")!
        functionNormalSmooth = defaultLibrary.makeFunction(name: "smoothNormal")!

        do {
            pipelineStateClothSim = try device.makeComputePipelineState(function: functionClothSim)
            pipelineStateNormalUpdate = try device.makeComputePipelineState(function: functionNormalUpdate)
            pipelineStateNormalSmooth = try device.makeComputePipelineState(function: functionNormalSmooth)
        } catch {
            fatalError("\(error)")
        }
    }
    @MainActor
    func createFlagSimulationFromNode(_ flag: ModelEntity) {
        let meshData = ClothSimMetalNode(device: device, width: width, height: height)
        
        guard let flagModel = flag.model else {return}
        
        guard let mesh = meshData.generateMeshResource() else { return }
        let clothEntity = ModelEntity(mesh: mesh)
        
        let boundingBox = flagModel.mesh.bounds
        let existingFlagBV = boundingBox.max - boundingBox.min
        let rescaleToMatchSizeMatrix = float4x4(existingFlagBV.x / Float(width))
        let rotation = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
        let localTransform = rescaleToMatchSizeMatrix * float4x4(rotation)
        
        clothEntity.transform.matrix = flag.transform.matrix * localTransform
        clothEntity.model?.materials = flagModel.materials

        flag.parent?.addChild(clothEntity)
        flag.removeFromParent()
        
        clothData.append(ClothData(clothEntity: clothEntity, meshData: meshData) )
    }
    
    func update(_ modelEntity: ModelEntity) {
        for cloth in clothData {
            let wind = SIMD3<Float>(1.8, 0.0, 0.0)
            
            // The multiplier is to rescale ball to flag model space.
            // The correct value should be passed in.
            let simData = SimulationData(wind: wind)
            deform(cloth.meshData, simData: simData)
        }
    }

    func deform(_ mesh: ClothSimMetalNode, simData: SimulationData) {
        var simData = simData
        
        let w = pipelineStateClothSim.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)
        
        let threadgroupsPerGrid = MTLSize(width: (mesh.vertexCount + w - 1) / w,
                                          height: 1,
                                          depth: 1)
        
        let clothSimCommandBuffer = commandQueue.makeCommandBuffer()
        let clothSimCommandEncoder = clothSimCommandBuffer?.makeComputeCommandEncoder()
        
        clothSimCommandEncoder?.setComputePipelineState(pipelineStateClothSim)
        
        clothSimCommandEncoder?.setBuffer(mesh.vb1, offset: 0, index: 0)
        clothSimCommandEncoder?.setBuffer(mesh.vb2, offset: 0, index: 1)
        clothSimCommandEncoder?.setBuffer(mesh.velocityBuffers[mesh.currentBufferIndex], offset: 0, index: 2)
        mesh.currentBufferIndex = (mesh.currentBufferIndex + 1) % 2
        clothSimCommandEncoder?.setBuffer(mesh.velocityBuffers[mesh.currentBufferIndex], offset: 0, index: 3)
        clothSimCommandEncoder?.setBytes(&simData, length: MemoryLayout<SimulationData>.size, index: 4)
        
        clothSimCommandEncoder?.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        clothSimCommandEncoder?.endEncoding()
        clothSimCommandBuffer?.commit()
        
        //
        
        let normalComputeCommandBuffer = commandQueue.makeCommandBuffer()
        let normalComputeCommandEncoder = normalComputeCommandBuffer?.makeComputeCommandEncoder()
        
        normalComputeCommandEncoder?.setComputePipelineState(pipelineStateNormalUpdate)
        normalComputeCommandEncoder?.setBuffer(mesh.vb2, offset: 0, index: 0)
        normalComputeCommandEncoder?.setBuffer(mesh.vb1, offset: 0, index: 1)
        normalComputeCommandEncoder?.setBuffer(mesh.normalWorkBuffer, offset: 0, index: 2)
        normalComputeCommandEncoder?.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        normalComputeCommandEncoder?.endEncoding()
        normalComputeCommandBuffer?.commit()

        //
        
        let normalSmoothComputeCommandBuffer = commandQueue.makeCommandBuffer()
        let normalSmoothComputeCommandEncoder = normalSmoothComputeCommandBuffer?.makeComputeCommandEncoder()
        
        normalSmoothComputeCommandEncoder?.setComputePipelineState(pipelineStateNormalSmooth)
        normalSmoothComputeCommandEncoder?.setBuffer(mesh.normalWorkBuffer, offset: 0, index: 0)
        normalSmoothComputeCommandEncoder?.setBuffer(mesh.normalBuffer, offset: 0, index: 1)
        normalSmoothComputeCommandEncoder?.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        normalSmoothComputeCommandEncoder?.endEncoding()
        normalSmoothComputeCommandBuffer?.commit()
    }
}
//#endif

