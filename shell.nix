{
  pkgs,
  withIDE ? (builtins.getEnv "WITH_IDE" == "1" || builtins.getEnv "WITH_IDE" == "true"),
}: let
  pname = "rustyvulkan";

  # Create a custom vulkan-validation-layers package built with gcc13
  customVulkanValidationLayers = pkgs.vulkan-validation-layers.override {
    stdenv = pkgs.gcc13Stdenv;
  };
  vsextensions =
    (with pkgs.vscode-extensions; [
      redhat.vscode-yaml
      vscodevim.vim
      github.copilot

      mikestead.dotenv
      naumovs.color-highlight
      oderwat.indent-rainbow
      streetsidesoftware.code-spell-checker
      vincaslt.highlight-matching-tag
      vscode-icons-team.vscode-icons

      timonwong.shellcheck
      foxundermoon.shell-format
      kamadorueda.alejandra # Nix code formatter
      jnoortheen.nix-ide

      rust-lang.rust-analyzer

      continue.continue # Local LLMs
    ])
    ++ pkgs.vscode-utils.extensionsFromVscodeMarketplace [
      {
        publisher = "kisstkondoros";
        name = "vscode-codemetrics";
        version = "1.26.1";
        sha256 = "sha256-lw6eZwlMXEjaT+FhhmiLkCB49Q7C015vU7zOLLTtGf8=";
      }
      {
        publisher = "ryanluker";
        name = "vscode-coverage-gutters";
        version = "2.13.0";
        sha256 = "sha256-qgPKGikqNIeZkKfd0P0keAdxRl9XNzvEJKQy58eaUZk=";
      }
      {
        publisher = "yzhang";
        name = "markdown-all-in-one";
        version = "3.6.3";
        sha256 = "sha256-xJhbFQSX1DDDp8iE/R8ep+1t5IRusBkvjHcNmvjrboM=";
      }
      {
        publisher = "dustypomerleau";
        name = "rust-syntax";
        version = "0.6.1";
        sha256 = "sha256-o9iXPhwkimxoJc1dLdaJ8nByLIaJSpGX/nKELC26jGU=";
      }
    ];
  rustyvulkanVsCodeWithExtensions = pkgs.vscode-with-extensions.override {
    vscodeExtensions = vsextensions;
  };
  vscodeWrapperContent = builtins.readFile "${rustyvulkanVsCodeWithExtensions}/bin/code";
  vscodeExtensionsDir = builtins.match ".*--extensions-dir ([^ ]+).*" vscodeWrapperContent;
  rustyvulkanCursorArgs = [
    "--extensions-dir ${builtins.elemAt vscodeExtensionsDir 0}"
    "--disable-extension Continue.continue"
    "--disable-extension kilocode.Kilo-Code"
  ];
  rustyvulkanCursorWithExtensions = pkgs.code-cursor.override {
    commandLineArgs = builtins.concatStringsSep " " rustyvulkanCursorArgs;
  };
  rustyvulkanVsCode = pkgs.writeShellScriptBin "rustyvulkan-vscode" ''
    ${rustyvulkanVsCodeWithExtensions}/bin/code --password-store="gnome-libsecret" "$@"
  '';
  rustyvulkanCursor = pkgs.writeShellScriptBin "rustyvulkan-cursor" ''
    ${rustyvulkanCursorWithExtensions}/bin/cursor --password-store="gnome-libsecret" "$@"
  '';
in
pkgs.mkShell {
  # Build-time dependencies (tools needed during compilation)
  nativeBuildInputs = with pkgs; [
    pkg-config
    fontconfig
    shaderc     # GLSL compiler
    glslang     # Shader validation and compilation
  ];

  # Runtime dependencies (libraries needed by the application)
  buildInputs = with pkgs; [
    vulkan-headers
    vulkan-loader
    customVulkanValidationLayers  # Use our custom-built validation layers
    vulkan-tools

    # X11 dependencies
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi

    libxkbcommon
  ];

  # Development tools available in shell
  packages = 
    with pkgs; ([
      rustup
      code-cursor
      blender
    ] ++ lib.optionals withIDE [
      # For VSCode and Cursor integration.
      rustyvulkanCursor
      rustyvulkanVsCode
    ]
  );

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
      fi
    else
      echo "Warning: vulkaninfo not found"
    fi
  '';
}
