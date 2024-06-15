//
//  ImmersiveView.swift
//  RealityComputeShader_Flag
//
//  Created by 许同学 on 2024/6/5.
//

import SwiftUI
import RealityKit

@MainActor
struct ImmersiveView: View {
    @State private var flagSimulation: MetalClothSimulator?
    @State private var subscriptions = [EventSubscription]()
    
    var body: some View {
        RealityView { content in
//            let material = SimpleMaterial(color: .red, isMetallic: false)
            let tex = try! await TextureResource.init(named: "LetsVisionOS")
            var material = PhysicallyBasedMaterial()
            material.baseColor = .init(texture: .init(tex))
            material.metallic = .init(floatLiteral: 0.6)
            
            let plane = MeshResource.generatePlane(width: 1, depth: 1)
            let entity = ModelEntity(mesh: plane, materials: [material])
            entity.position = simd_float3(x: -0.5, y: 0.8, z: -1.5)
            entity.name = "flag"
            content.add(entity)
            
            
            
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError( "Failed to get the system's default Metal device." )
            }
            flagSimulation = MetalClothSimulator(device: device)
            flagSimulation?.createFlagSimulationFromNode(entity)
            
            let sub = content.subscribe(to: SceneEvents.Update.self) { event in
                flagSimulation?.update()
            }
            subscriptions.append(sub)
        }
    }
}

#Preview(immersionStyle: .mixed) {
    ImmersiveView()
}
