{nixpkgs ? import <nixos-unstable> { }, cargo ? nixpkgs.cargo }:

with nixpkgs;

pkgs.stdenv.mkDerivation {
  name = "rustyvulkan";
  buildInputs = [ cargo python3 ];
}
