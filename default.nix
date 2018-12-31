{nixpkgs ? import <nixos-unstable> { } }:

with nixpkgs;

pkgs.stdenv.mkDerivation {
  name = "rustyvulkan";
  buildInputs = [ cargo cmake python3 vulkan-headers vulkan-loader vulkan-tools ];
  shellHook = ''
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.nix-profile/lib/"
  '';
}
