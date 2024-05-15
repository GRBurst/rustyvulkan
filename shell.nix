{ pkgs ? import <nixpkgs> { } }:
let
  pname = "rustyvulkan";
in
pkgs.mkShell {

  nativeBuildInputs = with pkgs; [
    pkg-config
    fontconfig
  ];

  buildInputs = with pkgs; [
    cargo

    vulkan-headers
    vulkan-loader
    vulkan-tools
    vulkan-validation-layers

    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi

    libxkbcommon

    shaderc
    glslang

    blender
  ];

  shellHook = ''
    echo "--- Welcome to ${pname}! ---"
    export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.nix-profile/lib/:${pkgs.vulkan-loader}/lib:${pkgs.libxkbcommon}/lib:${pkgs.vulkan-validation-layers}/lib"
  '';
}
