let
  pkgs = import <nixpkgs> { };
  pname = "rustyvulkan";
in
pkgs.mkShell {

  buildInputs = with pkgs; [
    cargo
    cmake
    python3
    vulkan-headers
    vulkan-loader
    vulkan-tools
  ];

  LD_LIBRARY_PATH="";

  shellHook = ''
    echo "--- Welcome to ${pname}! ---"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.nix-profile/lib/:${pkgs.vulkan-loader}/lib"
  '';
}