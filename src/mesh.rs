use std::{
    io::{BufRead, BufReader},
    mem::offset_of,
};

use bytemuck::{Pod, Zeroable};

use crate::errors::CrystalResult;

pub struct Attribute {
    pub size: usize,
    pub offset: usize,
}

type Vec3 = [f32; 3];
type Vec2 = [f32; 2];
pub type Index = u32;

// #[repr(C, align(16))]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexTexture {
    pub pos: Vec3,
    pub nor: Vec3,
    pub uv: Vec2,
    pub col: Vec3,
}

impl VertexTexture {
    pub fn get_attributes() -> Vec<Attribute> {
        vec![
            Attribute {
                size: size_of::<Vec3>(),
                offset: offset_of!(Self, pos),
            },
            Attribute {
                size: size_of::<Vec3>(),
                offset: offset_of!(Self, nor),
            },
            Attribute {
                size: size_of::<Vec2>(),
                offset: offset_of!(Self, uv),
            },
            Attribute {
                size: size_of::<Vec3>(),
                offset: offset_of!(Self, col),
            },
        ]
    }
}

pub struct Mesh {
    pub(crate) vertices: Vec<VertexTexture>,
    pub(crate) indices: Vec<Index>,
}

impl Mesh {
    pub fn from_buffer<T>(buffer: BufReader<T>) -> CrystalResult<Self>
    where
        BufReader<T>: BufRead,
    {
        let mut vertices = vec![];
        let mut indices = vec![];
        let mut normals: Vec<[f32; 3]> = vec![];
        let mut uvs: Vec<[f32; 2]> = vec![];

        for line in buffer.lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => continue,
            };

            let splitted: Vec<&str> = line.split_whitespace().collect();

            if splitted.len() == 0 || splitted[0].chars().next().unwrap() == '#' {
                continue;
            }

            if splitted.len() >= 3 {
                match splitted[0] {
                    "vn" => normals.push([
                        splitted[1].parse().unwrap(),
                        splitted[2].parse().unwrap(),
                        splitted[3].parse().unwrap(),
                    ]),
                    "vt" => uvs.push([splitted[1].parse().unwrap(), splitted[2].parse().unwrap()]),
                    "v" => vertices.push(VertexTexture {
                        pos: [
                            splitted[1].parse().unwrap(),
                            splitted[2].parse().unwrap(),
                            splitted[3].parse().unwrap(),
                        ],
                        nor: [0., 0., 0.],
                        uv: [0., 0.],
                        col: if splitted.len() != 7 {
                            [0., 0., 0.]
                        } else {
                            [
                                splitted[4].parse().unwrap_or(0.),
                                splitted[5].parse().unwrap_or(0.),
                                splitted[6].parse().unwrap_or(0.),
                            ]
                        },
                    }),
                    "f" => {
                        let mut local_indices = vec![];

                        for &data in &splitted[1..] {
                            if data.chars().next().unwrap() == '#' {
                                break;
                            }
                            let splitted: Vec<&str> = data.split('/').collect();

                            let idx: i32 = splitted[0].parse().unwrap();
                            let idx: Index = if idx >= 0 {
                                (idx - 1) as Index
                            } else {
                                (idx + vertices.len() as i32) as Index
                            };

                            if splitted[1].len() > 0 {
                                let uv: i32 = splitted[1].parse().unwrap();
                                let uv: usize = if uv >= 0 {
                                    (uv - 1) as usize
                                } else {
                                    (uv + uvs.len() as i32) as usize
                                };
                                vertices[idx as usize].uv = uvs[uv];
                            }

                            if splitted.len() > 2 && splitted[2].len() > 0 {
                                let nor: i32 = splitted[2].parse().unwrap();
                                let nor: usize = if nor >= 0 {
                                    (nor - 1) as usize
                                } else {
                                    (nor + normals.len() as i32) as usize
                                };
                                vertices[idx as usize].nor = normals[nor];
                            }

                            local_indices.push(idx);
                        }

                        if local_indices.len() == 3 {
                            indices.append(&mut local_indices);
                        } else if local_indices.len() == 4 {
                            indices.push(local_indices[0]);
                            indices.push(local_indices[1]);
                            indices.push(local_indices[2]);

                            indices.push(local_indices[0]);
                            indices.push(local_indices[2]);
                            indices.push(local_indices[3]);
                        }
                    }

                    _ => (),
                }
            }
        }

        Ok(Self { vertices, indices })
    }
}
