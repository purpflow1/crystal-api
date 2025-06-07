use std::{fs, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders/");

    let shader_dir = Path::new("shaders/");
    let out_dir = Path::new("shaders/");

    compile_shaders(shader_dir, &out_dir);
}

fn compile_shaders(shader_dir: &Path, out_dir: &Path) {
    let extensions = ["vert", "frag", "comp"];
    let mut compiled = false;

    for entry in fs::read_dir(shader_dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    compile_shader(&path, out_dir);
                    compiled = true;
                }
            }
        }
    }

    if !compiled {
        println!("cargo:warning=No shaders found in {}", shader_dir.display());
    }
}

fn compile_shader(shader_path: &Path, out_dir: &Path) {
    let file_name = shader_path.file_name().unwrap().to_str().unwrap();
    let output_path = out_dir.join(format!("{}.spv", file_name));

    let status = Command::new("glslc")
        .arg("-DDEBUG")
        .arg("-O")
        .arg(shader_path)
        .arg("-o")
        .arg(&output_path)
        .status()
        .unwrap_or_else(|_| panic!("Failed to execute glslc for {}", file_name));

    if !status.success() {
        panic!("Shader compilation failed for {}", file_name);
    }

    println!("cargo:rerun-if-changed={}", shader_path.display());
    println!("cargo:warning=Compiled shader: {}", file_name);
}
