# Vulkan Engine Refactoring Plan

This document outlines a step-by-step approach to refactoring the existing Vulkan engine codebase. The goal is to move code from the monolithic `main.rs` into appropriate modules while ensuring functionality is preserved throughout the process.

## Guiding Principles

- **Incremental Changes**: Refactor one component at a time
- **Continuous Verification**: Test functionality after each step to catch regressions early
- **Bottom-Up Approach**: Start with self-contained, low-level components
- **Clear Dependencies**: Ensure proper module relationships

## Phase 1: Extract Low-Level Components

### Step 1: Extract Vulkan Device/Pipeline Logic
- Move pipeline creation code to `src/renderer/pipeline.rs`
- Extract shader module creation functions
- Move pipeline layout creation code
- Create proper abstractions for pipeline state
- **Verification**: Ensure rendering still works with extracted pipeline code

### Step 2: Extract Resource Management
- Complete `src/resources/texture.rs` implementation 
- Move texture creation and loading code from main.rs
- Implement proper resource loading/unloading
- **Verification**: Confirm textures still load and render correctly

### Step 3: Extract Buffer Management
- Create a new `src/renderer/buffer.rs` module
- Move vertex, index, and uniform buffer creation code
- Create abstractions for different buffer types
- Implement memory management utilities
- **Verification**: Ensure models still load and render correctly

## Phase 2: Extract Mid-Level Systems

### Step 4: Extract Input Handling
- Implement `src/platform/input.rs`
- Create an input system to handle keyboard and mouse input
- Move cursor capturing and mouse movement code
- Implement input mapping system
- **Verification**: Test all controls to ensure they work as before

### Step 5: Extract Window Management
- Implement `src/platform/window.rs`
- Create window management abstractions
- Move window creation and event handling code
- Handle resize events and window state
- **Verification**: Test window resizing and full-screen toggling

### Step 6: Extract Rendering Loop
- Create `src/renderer/render_loop.rs`
- Move frame rendering and synchronization code
- Implement proper frame timing
- Extract swapchain recreation logic
- **Verification**: Ensure rendering still works with correct frame pacing

## Phase 3: Extract High-Level Systems

### Step 7: Complete GameObject Systems
- Enhance `src/scene/gameobject.rs`
- Implement component system if needed
- Move GameObject initialization and management
- Extract model loading and transformation code
- **Verification**: Ensure all game objects render correctly with transformations

### Step 8: Implement Game State Management
- Implement `src/game/game_state.rs` and related files
- Create a state machine for application flow
- Implement basic states (loading, running, paused)
- Move relevant code from main loop
- **Verification**: Test state transitions and ensure proper game flow

### Step 9: Create Engine Core
- Implement `src/core/engine.rs`
- Create a central engine class to manage all systems
- Move initialization and shutdown logic
- Implement update and render loops
- **Verification**: Ensure the application runs correctly through the engine abstraction

## Phase 4: Final Refactoring

### Step 10: Clean up main.rs
- Reduce main.rs to only engine initialization
- Move configuration to `src/core/config.rs`
- Implement command-line argument parsing
- **Verification**: Ensure the application starts correctly with minimal main.rs

### Step 11: Add Error Handling
- Replace unwrap() calls with proper error handling
- Implement custom error types if needed
- Add logging for errors and warnings
- **Verification**: Test error cases to ensure they're handled gracefully

### Step 12: Add Documentation
- Document public APIs
- Add usage examples
- Create architecture documentation
- Document the rendering pipeline
- **Verification**: Ensure documentation is clear and up to date

## Verification Strategy

After each refactoring step:

1. **Check, Compile and Run**: Ensure the application compiles and runs without errors
    - run `cargo check`
    - run `cargo run` and check if the applications works
2. **Functional Testing**: Test all features that might be affected by the changes
   - Camera movement and rotation
   - Object rendering
   - Texture loading
   - Window resizing
   - Input handling
3. **Performance Check**: Ensure no significant performance regressions
4. **Code Review**: Review the refactored code for clarity and maintainability

## Progress Tracking

| Step | Description | Status | Date Completed | Notes |
|------|-------------|--------|----------------|-------|
| 1    | Extract Pipeline Logic | Completed | 2024-07-15 | Moved pipeline creation, shader loading/creation to `src/renderer/pipeline.rs` with proper documentation. Application verified working. |
| 2    | Extract Resource Management | Completed | 2024-07-15 | Refactored texture handling code to `src/resources/texture.rs`, including image loading, texture creation, depth/color textures, and memory management. Fixed memory type lookup to match the original implementation exactly. |
| 3    | Extract Buffer Management | Completed | 2024-07-15 | Created `src/renderer/buffer.rs` module including vertex, index, and uniform buffer management. Made buffer functions public when needed by texture module. Added proper documentation for each function. |
| 4    | Extract Input Handling | Completed | 2024-07-15 | Implemented `src/platform/input.rs` with a dedicated InputSystem class for keyboard, mouse, and cursor handling. Updated main.rs to use the new system. Verified input functionality works correctly for camera and movement. |
| 5    | Extract Window Management | Completed | 2024-07-15 | Implemented `src/platform/window.rs` with a dedicated WindowSystem class. Also implemented resource manager to fix flickering issues during swapchain recreation, ensuring proper texture reuse and descriptor set recreation. Improved resource lifecycle management during window resize events. |
| 6    | Extract Rendering Loop | In Progress | | Created `src/renderer/render_loop.rs` with a dedicated RenderLoop class for frame rendering, synchronization, and swapchain management. Implemented core functionality like draw_frame, update_uniform_buffers, and create_swapchain methods, following original code closely. Fixed compilation errors and ensured proper queue family handling. Next steps: implement remaining helper methods and update main.rs to use the new system. |
| 7    | Complete GameObject Systems | Not Started | | |
| 8    | Implement Game State | Not Started | | |
| 9    | Create Engine Core | Not Started | | |
| 10   | Clean up main.rs | Not Started | | |
| 11   | Add Error Handling | Not Started | | |
| 12   | Add Documentation | Not Started | | | 

## Refactoring Learnings

### Learnings from Step 1: Extract Pipeline Logic
- **Focused Changes**: Keeping changes narrowly focused on only the targeted functionality helps minimize the risk of introducing bugs.
- **Module Structure**: Having a clear module structure with well-defined responsibilities makes the refactoring process smoother.
- **Documentation**: Adding proper documentation during refactoring improves code maintainability and helps clarify function purposes.
- **Verification Importance**: The verification process after each step is crucial to ensure no regressions are introduced.

### Learnings from Step 2: Extract Resource Management
- **Exact Logic Duplication**: When refactoring low-level functionality like memory allocation, it's critical to exactly match the original implementation logic to avoid subtle bugs.
- **Function Dependencies**: Understanding inter-function dependencies is essential when refactoring - we had to refactor memory management and buffer allocation together with texture handling.
- **API Selection**: Using the right API methods is crucial - we encountered issues when using `memory_properties()` instead of `get_mem_properties()` despite them serving a similar purpose.
- **Error Messages**: Detailed error messages helped pinpoint the exact location of issues, making debugging faster and more focused.

### Learnings from Step 3: Extract Buffer Management
- **Module Organization**: Using a modern Rust module approach (with individual files rather than mod.rs) helps maintain a cleaner directory structure.
- **Public vs Private APIs**: It's important to carefully consider which functions need to be public for use by other modules (like `find_memory_type`).
- **Dependency Management**: Buffer and texture handling have shared dependencies on memory management, showing the importance of proper abstraction boundaries.
- **Type Traits**: Adding derive attributes like `#[derive(Copy, Clone)]` for data structures is important for allowing operations like memory copying.

### Learnings from Step 4: Extract Input Handling
- **State Management**: Creating a dedicated input system allows for cleaner state management and separation of concerns.
- **Borrowing Rules**: Rust's borrowing rules require careful design when sharing data between components (like player and camera objects).
- **Functional Approach**: Using a more functional approach with well-defined state transitions helped avoid borrowing issues.
- **Declarative API**: Creating a declarative API for the input system makes it easier to use and understand by abstracting away implementation details.

### Learnings from Step 5: Extract Window Management
- **Minimizing Changes**: Focusing on minimal changes to existing architecture helps maintain stability during refactoring.
- **API Compatibility**: Window APIs can change between library versions, requiring adaptability (we had to update window handle APIs).
- **Avoiding Ownership Complexity**: Keeping ownership models simple avoids issues where components might try to own resources that need external management.
- **Progressive Refactoring**: Sometimes it's better to extract functionality without trying to redesign the entire architecture at once.
- **Error Handling**: Proper error handling is critical in platform-specific code like window management, as system-level failures can occur.
- **Asset Paths**: When refactoring code, it's important to ensure assets (like textures and models) can still be located correctly. The refactoring process helped us identify and fix an incorrect asset path.
- **Buffer Lifecycle Management**: When recreating resources like swapchains, it's critical to properly clean up all associated resources, including buffers in game objects. We found and fixed an issue where game object buffers weren't being properly cleaned up during swapchain recreation, leading to validation errors.
- **Remaining Resource Issues**: We still have some validation errors related to Vulkan objects not being destroyed at application exit. This suggests that there are additional resource lifecycle issues that will need to be addressed in a future step, possibly when implementing proper error handling (Step 11).
- **Resource Caching**: Implementing resource caching for frequently used assets like textures significantly improves performance and stability during window resizing.
- **Resource Reuse Strategy**: Preserving and reusing game objects and textures during swapchain recreation prevents flickering and visual artifacts.
- **Descriptor Set Management**: Always recreate descriptor sets after updating uniform buffers to maintain proper bindings between resources.
- **Lifecycle Dependencies**: Understanding resource dependencies (e.g., descriptor sets depending on buffers and textures) is crucial for proper recreation sequences.
- **Swapchain Recreation Optimization**: A proper swapchain recreation strategy should preserve static resources while recreating only what's necessary.
- **Resource Tracking for Cleanup**: Implementing a tracking system (like HashSets of handle identifiers) to prevent double-free issues is crucial for proper Vulkan resource cleanup. Without such tracking, validation errors can occur during application shutdown when attempting to free already destroyed resources. In a more comprehensive resource management system, a central registry of active resources would be maintained to ensure each resource is only freed once.
- **Defensive Programming with Null Checks**: Adding null handle checks before destroying resources is essential in Vulkan applications to avoid validation errors. Checking `handle.as_raw() != 0` helps prevent attempting to destroy already-freed resources.
- **Handle Invalidation**: After destroying a Vulkan resource, setting its handle to null (e.g., `handle = vk::Buffer::null()`) prevents accidental reuse and double-free issues.
- **Mutable Iteration**: Using `iter_mut()` instead of `iter()` when cleaning up resources allows setting handles to null after destruction, improving safety.
- **Explicit Descriptor Management**: Clear descriptor set lists after resetting the descriptor pool to prevent stale references.
- **Validation Layer Messages**: Carefully analyze validation layer messages - they often point to deeper architectural issues in resource management rather than simple bugs.
- **Texture Resource Management**: Special care is needed for textures that contain multiple Vulkan objects (image, memory, view, sampler). Each must be properly nullified after destruction.

### Learnings from Step 6: Extract Rendering Loop
- **Modular Design**: Separating the rendering loop logic into its own module clarifies responsibilities and improves maintainability.
- **Resource Lifecycle Management**: The refactoring process highlighted the need for careful management of resource lifecycles, particularly around swapchain recreation.
- **Synchronization Complexity**: Vulkan's synchronization mechanisms (semaphores, fences) require careful handling to ensure correct frame pacing and resource access.
- **API Abstraction**: Creating higher-level abstractions around the Vulkan API makes the code more readable and easier to maintain.
- **Resource Dependencies**: The rendering loop has complex dependencies on various resources (swapchain, pipelines, command buffers) that need to be managed carefully.
- **Forward References**: To maintain the existing architecture, many method stubs were created to be filled in later, showing how step-by-step refactoring works in a large codebase.
- **Architecture Planning**: Refactoring the rendering loop required planning for how different components will interact, showing the importance of architecture design in refactoring.
- **Shared Resources**: Using direct references instead of Arc improves ownership semantics and reflects the original code's structure.
- **Code Duplication**: In this refactoring phase, we focused on correctly duplicating the original logic rather than optimizing or restructuring it, ensuring compatibility with existing code.
- **Exact Logic Duplication**: When refactoring critical functions like swapchain creation, we needed to exactly match the original code to avoid subtle errors. 