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
| 5    | Extract Window Management | Not Started | | |
| 6    | Extract Rendering Loop | Not Started | | |
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