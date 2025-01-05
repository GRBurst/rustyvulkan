{
  pkgs,
}: let
  pname = "rustyvulkan";
in
pkgs.mkShell {

  nativeBuildInputs = with pkgs; [
    pkg-config
    fontconfig
  ];

  buildInputs = with pkgs; [
    rustup

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

    export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.nix-profile/lib/:${pkgs.vulkan-loader}/lib:${pkgs.libxkbcommon}/lib:${pkgs.vulkan-validation-layers}/lib"
  '';
}
