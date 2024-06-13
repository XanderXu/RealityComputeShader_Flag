//
//  RealityComputeShader_FlagV2App.swift
//  RealityComputeShader_FlagV2
//
//  Created by 许同学 on 2024/6/13.
//

import SwiftUI

@main
struct RealityComputeShader_FlagV2App: App {

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
