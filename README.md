# Crystal API
**Crystal API**  is a sub-high level graphics wrapper for graphics API. Currently in WIP stage, but usable for now.

## Features:
- Vulkan API support
- Synchronized in-frame and out-of-frame compute operations
- Direct buffer access (no staging in buffer operations)
- Multithreaded access to resource
- Unified app architecture

## Long-term goals
- Adding _non-present_ **Crystal API** entry (for compute operations only)
- Adding **Metal API** support
- Adding **DirectX** support

## Short-term goals
- Adding raytracing support

---

## Installing
```bash
cargo add crystal-api
```

## Running
To run debug scene from included example use:
```bash
cargo run --manifest-path=examples/array-load-test/Cargo.toml
```
_!!! Khronos validation layers are required to run debug builds._
