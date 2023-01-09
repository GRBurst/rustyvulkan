let
  pkgs = import <nixpkgs> { };
  pname = "rustyvulkan";
in
pkgs.mkShell {

  nativeBuildInputs = with pkgs; [
    pkg-config
    fontconfig
  ];

  buildInputs = with pkgs; [
    cargo
    cmake
    python3
    vulkan-headers
    vulkan-loader
    vulkan-tools
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
  ];

  shellHook = ''
    echo "--- Welcome to ${pname}! ---"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.nix-profile/lib/:${pkgs.vulkan-loader}/lib"
  '';
}
