//
//  RealityComputeShader_FlagApp.swift
//  RealityComputeShader_Flag
//
//  Created by 许同学 on 2024/6/5.
//

import SwiftUI

@main
struct RealityComputeShader_FlagApp: App {
    @State private var appModel = AppModel()
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appModel)
        }

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            ImmersiveView()
                .environment(appModel)
                .onAppear {
                    appModel.immersiveSpaceState = .open
                }
                .onDisappear {
                    appModel.immersiveSpaceState = .closed
                }
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}
