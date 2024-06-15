//
//  ContentView.swift
//  RealityComputeShader_FlagV2
//
//  Created by 许同学 on 2024/6/13.
//

import SwiftUI
import RealityKit

struct ContentView: View {

    var body: some View {
        VStack {
            Text("Hello, Compute Shader!")
                .font(.title)

            Text("https://github.com/XanderXu/RealityComputeShader_Flag")
                .font(.title2)
                .foregroundStyle(.white)
                .underline(true,pattern: .solid)
                .padding(.bottom)

            ToggleImmersiveSpaceButton()
        }
        .padding()
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
