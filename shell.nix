{
  pkgs,
}: let
  pname = "rustyvulkan";
in
pkgs.mkShell {
  # Build-time dependencies (tools needed during compilation)
  nativeBuildInputs = with pkgs; [
    pkg-config
    fontconfig
    shaderc
    glslang
  ];

  # Runtime dependencies (libraries needed by the application)
  buildInputs = with pkgs; [
    vulkan-headers
    vulkan-loader
    vulkan-validation-layers
    vulkan-tools

    # X11 dependencies
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    
    libxkbcommon
  ];

  # Development tools available in shell
  packages = with pkgs; [
    rustup
    code-cursor
    blender
  ];

  shellHook = ''
    echo "--- Welcome to ${pname}! ---"
    projectDir=$PWD
    rustupHomeDir="$projectDir/.rustup"
    mkdir -p "$rustupHomeDir"

    export RUSTUP_HOME="$rustupHomeDir"
    export LIBRARY_PATH="$LIBRARY_PATH:$projectDir/nix/profile/default/lib"

    # Vulkan configuration
    export ENABLE_VALIDATION_LAYERS=1
    export VK_LAYER_PATH="${customVulkanValidationLayers}/share/vulkan/explicit_layer.d"
    export VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"

    # Make sure both runtime and validation layers are in library path
    export LD_LIBRARY_PATH="${customVulkanValidationLayers}/lib:${pkgs.vulkan-loader}/lib:${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH"

    # Additional Vulkan debug settings
    export VK_LOADER_DEBUG=all
    export VK_LOADER_LAYERS_ENABLE=1

    # Verify validation layers are available
    echo "Checking Vulkan validation layers..."
    if command -v vulkaninfo >/dev/null 2>&1; then
      if vulkaninfo 2>/dev/null | grep -q "VK_LAYER_KHRONOS_validation"; then
        echo "Vulkan validation layers are properly installed"
      else
        echo "Warning: Vulkan validation layers not found in vulkaninfo"
        echo "Validation layer path: $VK_LAYER_PATH"
        ls -l "$VK_LAYER_PATH" || echo "Directory not found!"
        echo "You might have version missmatches. Try nix-garbage-collect and re-enter the shell."
      fi
    else
      echo "Warning: vulkaninfo not found"
    fi
  '';
}
