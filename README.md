# Crystal API
**Crystal API**  is a sub-high level graphics wrapper for graphics API. Currently in WIP stage, but usable for now.

## Features:
- Vulkan API support
- Synchronized in-frame and out-of-frame compute operations
- Direct buffer access (no staging in buffer operations)
- Multithreaded access to resources

## Long-term goals
- Adding _non-present_ **Crystal API** entry (for compute operations only)
- Adding **Metal API** support
- Adding **DirectX** support

## Short-term goals
- Adding raytracing support

---

## Installing
There is no cargo crate for this repo for now, so you need to clone it:
```bash
git clone https://github.com/purpflow1/crystal-api.git
```

## Running
First of all, `cd` to the cloned directory:
```bash
cd crystal-api/
```

To run debug scene from included example use:
```bash
cargo run --example main
```
_!!! Khronos validation layers are required to run debug builds._

To build release _rlib_ use:
```bash
cargo build --release
```
