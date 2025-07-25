# Crystal API
**Crystal API**  is a sub-high level graphics wrapper for graphics API. Currently in WIP stage, but usable for now.

## Features:
- Vulkan API support
- Synchronized in-frame and out-of-frame compute operations
- Direct buffer access (no staging in buffer operations)
- Multithreaded access to resource
- Unified app architecture

## Goals
- Adding **Metal API** support
- Adding **DirectX** support
- Adding raytracing support

---

## Running examples
To run debug scene from included example in
[github](https://github.com/purpflow1/crystal-api) use:
```bash
cargo run --manifest-path=examples/array-load-test/Cargo.toml
```
_Khronos validation layers are required to run debug builds!_
_Visit https://vulkan.lunarg.com._
