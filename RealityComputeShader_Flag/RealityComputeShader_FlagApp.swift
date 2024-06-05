//
//  RealityComputeShader_FlagApp.swift
//  RealityComputeShader_Flag
//
//  Created by 许同学 on 2024/6/5.
//

import SwiftUI

@main
struct RealityComputeShader_FlagApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }

        ImmersiveSpace(id: "ImmersiveSpace") {
            ImmersiveView()
        }
    }
}
