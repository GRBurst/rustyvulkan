{
  description = "Custom Vulkan SDK";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        vulkanComponents = rec {
          validation-layers = pkgs.vulkan-validation-layers.override {
            stdenv = pkgs.gcc13Stdenv;
          };
          
          headers = pkgs.vulkan-headers;
          loader = pkgs.vulkan-loader.override {
            stdenv = pkgs.gcc13Stdenv;
          };
          tools = pkgs.vulkan-tools.override {
            stdenv = pkgs.gcc13Stdenv;
          };
          
          # Shader compilation tools
          shaderc = pkgs.shaderc.override {
            stdenv = pkgs.gcc13Stdenv;
          };
          glslang = pkgs.glslang.override {
            stdenv = pkgs.gcc13Stdenv;
          };
        };

        vulkan-sdk = pkgs.symlinkJoin {
          name = "vulkan-sdk";
          paths = with vulkanComponents; [
            validation-layers
            headers
            loader
            tools
            shaderc
            glslang
          ];

          setupHook = pkgs.writeText "setup-hook.sh" ''
            export VULKAN_SDK="@out@"
            export VK_LAYER_PATH="@out@/share/vulkan/explicit_layer.d"
            export LD_LIBRARY_PATH="@out@/lib:''${LD_LIBRARY_PATH:-}"
            export VK_LOADER_DEBUG=all
            export VK_LOADER_LAYERS_ENABLE=1
            export ENABLE_VALIDATION_LAYERS=1
            export VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
          '';
        };

      in {
        packages.default = vulkan-sdk;
      });
} 