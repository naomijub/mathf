use crate::math_helper;
use crate::vector::Vector;

use super::matrix::Matrix3x3 as M;
use std::ops;

///A 3D Vector with x, y, z coordinates: Vector3
#[derive(Clone, PartialEq, Debug)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

///A 3D Point with x, y and z coordinates: Point3
#[derive(PartialEq, Debug)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    ///Instantiates a new vector with to be defined values of x, y, z;
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x: x, y: y, z: z }
    }

    ///Instantiates a new Vector3 from 2 Point3 (initial position, final position).
    ///The new vector is created as final - initial (Points)
    pub fn diff(origin: Point3, destination: Point3) -> Vector3 {
        Vector3 {
            x: destination.x - origin.x,
            y: destination.y - origin.y,
            z: destination.z - origin.z,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// | 0 |
    /// | 1 |
    /// | 0 |
    /// y=1, x=0, z=0
    pub fn UP() -> Vector3 {
        Vector3 {
            x: 0f32,
            y: 1f32,
            z: 0f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// |  0 |
    /// | -1 |
    /// |  0 |
    /// y=-1, x=0, z=0
    pub fn DOWN() -> Vector3 {
        Vector3 {
            x: 0f32,
            y: -1f32,
            z: 0f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// | 1 |
    /// | 0 |
    /// | 0 |
    /// y=0, x=1, z=0
    pub fn RIGHT() -> Vector3 {
        Vector3 {
            x: 1f32,
            y: 0f32,
            z: 0f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// | -1 |
    /// |  0 |
    /// |  0 |
    /// y=0, x=-1, z=0
    pub fn LEFT() -> Vector3 {
        Vector3 {
            x: -1f32,
            y: 0f32,
            z: 0f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// | 0 |
    /// | 0 |
    /// | 1 |
    /// y=0, x=0, z=1
    pub fn FOWARD() -> Vector3 {
        Vector3 {
            x: 0f32,
            y: 0f32,
            z: 1f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// |  0 |
    /// |  0 |
    /// | -1 |
    /// y=0, x=0, z=-1
    pub fn BACK() -> Vector3 {
        Vector3 {
            x: 0f32,
            y: 0f32,
            z: -1f32,
        }
    }

    ///Transforms a Vector 3 from one vectorspace to another  via a matrix3x3 transform.
    /// Same as Vector2 but with Matrix3x3 and Vector3.
    pub fn transform(&self, m: M, vec: Vector3) -> Vector3 {
        (m * self) + vec
    }

    ///Cross product between two vectors 3.
    /// | a |   | m |   | do - gn |
    /// | d | x | n | = | gm - ao |
    /// | g |   | o |   | an - dm |
    pub fn x(&self, vec: Vector3) -> Vector3 {
        Vector3 {
            x: self.y * vec.z - self.z * vec.y,
            y: self.z * vec.x - self.x * vec.z,
            z: self.x * vec.y - self.y * vec.x,
        }
    }

    ///Scales a Vector 3 in a non uniform way: (a, b, c) * (x, y, z) = (ax, by, cz)
    pub fn nonuniform_scale(&self, a: f32, b: f32, c: f32) -> Vector3 {
        let scale_matrix = M::new(
            Vector3::new(a, 0f32, 0f32),
            Vector3::new(0f32, b, 0f32),
            Vector3::new(0f32, 0f32, c),
        );
        self.transform(scale_matrix, Vector3::ZERO())
    }
}

impl Vector for Vector3 {
    #[allow(dead_code, non_snake_case)]
    /// | 1 |
    /// | 1 |
    /// | 1 |
    ///x=1, y=1, z=1
    fn ONE() -> Vector3 {
        Vector3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        }
    }

    #[allow(dead_code, non_snake_case)]
    /// | 0 |
    /// | 0 |
    /// | 0 |
    ///Defines a Modulus ZERO Vector (x=0, y=0, z=0)
    fn ZERO() -> Vector3 {
        Vector3 {
            x: 0f32,
            y: 0f32,
            z: 0f32,
        }
    }

    ///Vector magnitude: the square root of the sum of each vector part to the power of 2
    fn magnitude(&self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
    }

    ///Transforms a Vector3 into a Vec<f32>
    fn to_vector(&self) -> Vec<f32> {
        vec![self.x, self.y, self.z]
    }
}

impl ops::Add for Vector3 {
    type Output = Vector3;

    ///Implements the Vector3 '+' trait
    fn add(self, new_vec: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + new_vec.x,
            y: self.y + new_vec.y,
            z: self.z + new_vec.z,
        }
    }
}

impl ops::Add for &Vector3 {
    type Output = Vector3;

    ///Implements the &Vector3 '+' trait
    fn add(self, new_vec: &Vector3) -> Vector3 {
        Vector3 {
            x: self.x + new_vec.x,
            y: self.y + new_vec.y,
            z: self.z + new_vec.z,
        }
    }
}

impl ops::Mul<f32> for Vector3 {
    type Output = Vector3;

    ///Implements the scalar multiplication of a Vector3 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn mul(self, value: f32) -> Vector3 {
        Vector3 {
            x: self.x * value,
            y: self.y * value,
            z: self.z * value,
        }
    }
}

impl ops::Mul<f32> for &Vector3 {
    type Output = Vector3;

    ///Implements the scalar multiplication of a &Vector3 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn mul(self, value: f32) -> Vector3 {
        Vector3 {
            x: self.x * value,
            y: self.y * value,
            z: self.z * value,
        }
    }
}

impl ops::Div<f32> for Vector3 {
    type Output = Vector3;

    ///Implements the scalar division of a Vector3 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn div(self, value: f32) -> Vector3 {
        Vector3 {
            x: self.x / value,
            y: self.y / value,
            z: self.z / value,
        }
    }
}

impl ops::Div<f32> for &Vector3 {
    type Output = Vector3;

    ///Implements the scalar division of a &Vector3 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn div(self, value: f32) -> Vector3 {
        Vector3 {
            x: self.x / value,
            y: self.y / value,
            z: self.z / value,
        }
    }
}

impl ops::Mul<Vector3> for Vector3 {
    type Output = f32;

    ///Implements the dot product of 2 Vector3 as '*'.
    fn mul(self, new_vec: Vector3) -> f32 {
        self.x * new_vec.x + self.y * new_vec.y + self.z * new_vec.z
    }
}

impl ops::Mul<&Vector3> for &Vector3 {
    type Output = f32;

    ///Implements the dot product of 2 &Vector3 as '*'.
    fn mul(self, new_vec: &Vector3) -> f32 {
        self.x * new_vec.x + self.y * new_vec.y + self.z * new_vec.z
    }
}

impl ops::Sub for Vector3 {
    type Output = Vector3;

    ///Implements the Vector3 '-' trait
    fn sub(self, new_vec: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - new_vec.x,
            y: self.y - new_vec.y,
            z: self.z - new_vec.z,
        }
    }
}

impl ops::Sub for &Vector3 {
    type Output = Vector3;

    ///Implements the &Vector3 '-' trait
    fn sub(self, new_vec: &Vector3) -> Vector3 {
        Vector3 {
            x: self.x - new_vec.x,
            y: self.y - new_vec.y,
            z: self.z - new_vec.z,
        }
    }
}

impl Point3 {
    ///Instantiates a new Point3 with x, y and z.
    pub fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3 { x: x, y: y, z: z }
    }

    ///Creates a new Vector3 relative to position (0, 0, 0)
    pub fn to_vec(self) -> Vector3 {
        Vector3::diff(Point3::origin(), self)
    }

    ///Instantiates a Point3 with (0, 0, 0)
    fn origin() -> Point3 {
        Point3::new(0f32, 0f32, 0f32)
    }

    #[allow(dead_code, non_snake_case)]
    ///Instantiates a Point3 with (1, 1, 1)
    fn ONE() -> Point3 {
        Point3::new(1f32, 1f32, 1f32)
    }

    ///Transforms a Point 3 from one vectorspace to another  via a matrix3x3 transform.
    /// Same as Point2 but with Point3, Matrix3x3 and Vector3.
    pub fn transform(&self, m: M, vec: Vector3) -> Point3 {
        (m * self) + vec
    }
}

impl ops::Add<Vector3> for Point3 {
    type Output = Point3;

    ///Overloads + for Points and Vectors: P + PQ = Q
    fn add(self, new_vec: Vector3) -> Point3 {
        Point3 {
            x: self.x + new_vec.x,
            y: self.y + new_vec.y,
            z: self.z + new_vec.z,
        }
    }
}

impl ops::Mul<&Point3> for &Vector3 {
    type Output = f32;

    ///Implements the dot product of &Point3 and &Vector3 as '*'.
    fn mul(self, new_vec: &Point3) -> f32 {
        self.x * new_vec.x + self.y * new_vec.y + self.z * new_vec.z
    }
}

// Indexing
use std::ops::Index;

impl Index<usize> for Vector3 {
    type Output = f32;
    fn index(&self, s: usize) -> &f32 {
        match s {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bonds"),
        }
    }
}

#[allow(dead_code)]
///Vector3 linear indenpendency (3D)
pub fn lin_ind(vec1: &Vector3, vec2: &Vector3, vec3: &Vector3, delta: f32) -> bool {
    let dot1 = vec1 * vec2;
    let dot2 = vec1 * vec3;
    let dot3 = vec3 * vec2;
    math_helper::float_eq(dot1, 0f32, delta)
        && math_helper::float_eq(dot2, 0f32, delta)
        && math_helper::float_eq(dot3, 0f32, delta)
}

#[allow(dead_code)]
/// Cos between two vector3
pub fn cos(vec1: &Vector3, vec2: &Vector3) -> f32 {
    let dot_product = vec1 * vec2;
    let denominator = vec1.magnitude() * vec2.magnitude();
    dot_product / denominator
}

#[allow(dead_code)]
/// Sin between two vector3
pub fn sin(vec1: &Vector3, vec2: &Vector3) -> f32 {
    let cos = cos(vec1, vec2);
    (1f32 - cos.powi(2)).sqrt()
}

#[allow(dead_code)]
///Distance between 2 point3
pub fn dist(a: &Point3, b: &Point3) -> f32 {
    let x_dist = (a.x - b.x).powi(2);
    let y_dist = (a.y - b.y).powi(2);
    let z_dist = (a.z - b.z).powi(2);
    (x_dist + y_dist + z_dist).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vector3_with_parameters() {
        let actual = Vector3::new(1f32, 1f32, 1f32);
        let expected = Vector3 {
            x: 1f32,
            y: 1f32,
            z: 1f32,
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn creates_vector3_up() {
        let actual = Vector3::UP();
        assert!(actual.x == 0f32 && actual.y == 1f32 && actual.z == 0f32);
    }

    #[test]
    fn adds_right_and_left_vectors() {
        let actual = Vector3::RIGHT() + Vector3::LEFT();
        assert_eq!(actual.x, 0f32);
    }

    #[test]
    fn adds_right_and_left_vectors_by_ref() {
        let actual = &Vector3::RIGHT() + &Vector3::LEFT();
        assert_eq!(actual.x, 0f32);
    }

    #[test]
    fn adds_right_and_up() {
        let actual = Vector3::RIGHT() + Vector3::UP();
        assert_eq!(actual.x, 1f32);
        assert_eq!(actual.y, 1f32);
        assert_eq!(actual.z, 0f32);
    }

    #[test]
    fn mult_one_by_3() {
        let actual = Vector3::ONE() * 3f32;
        assert_eq!(actual.x, 3f32);
        assert_eq!(actual.y, 3f32);
        assert_eq!(actual.z, 3f32);
    }

    #[test]
    fn mult_one_by_3_by_ref() {
        let actual = &Vector3::ONE() * 3f32;
        assert_eq!(actual.x, 3f32);
        assert_eq!(actual.y, 3f32);
        assert_eq!(actual.z, 3f32);
    }

    #[test]
    fn sub_right_from_one() {
        let actual = Vector3::ONE() - Vector3::RIGHT();
        assert_eq!(actual.x, 0f32);
        assert_eq!(actual.y, 1f32);
        assert_eq!(actual.z, 1f32);
    }

    #[test]
    fn sub_right_from_one_by_ref() {
        let actual = &Vector3::ONE() - &Vector3::RIGHT();
        assert_eq!(actual.x, 0f32);
        assert_eq!(actual.y, 1f32);
        assert_eq!(actual.z, 1f32);
    }

    #[test]
    fn magnitude_of_vector() {
        let vec = Vector3 {
            x: 1f32,
            y: 2f32,
            z: 3f32,
        };
        assert_eq!(vec.magnitude(), 3.7416575f32);
    }

    #[test]
    fn magnitude_of_vector_is_positive() {
        let vec = Vector3 {
            x: -3f32,
            y: 4f32,
            z: 5f32,
        };
        assert!(vec.magnitude() >= 0f32);
    }

    #[test]
    fn dot_product() {
        let vec1 = Vector3 {
            x: 2f32,
            y: 1f32,
            z: 2f32,
        };
        let vec2 = Vector3 {
            x: 1f32,
            y: 2f32,
            z: 3f32,
        };
        let actual = vec1 * vec2;
        let expected = 10f32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn constructs_vector3_from_points3() {
        let vec = Vector3::diff(
            Point3::new(1f32, -1f32, 2f32),
            Point3::new(2f32, 3f32, 2f32),
        );
        assert_eq!(vec.x, 1f32);
        assert_eq!(vec.y, 4f32);
        assert_eq!(vec.z, 0f32);
    }

    #[test]
    fn creates_vector_from_point3() {
        let point = Point3::new(1f32, 1f32, 1f32);
        let actual = point.to_vec();
        let expected = Vector3::ONE();
        assert_eq!(expected, actual);
    }

    #[test]
    fn point_add_vector_result_new_point() {
        let point = Point3::origin();
        let vec = Vector3::ONE();
        let actual = point + vec;
        assert_eq!(actual.x, 1f32);
        assert_eq!(actual.y, 1f32);
        assert_eq!(actual.z, 1f32);
    }

    #[test]
    fn cross_product_between_2_vectors() {
        let vec1 = Vector3::new(1f32, 2f32, 3f32);
        let actual = vec1.x(Vector3::new(2f32, 3f32, 4f32));
        let expected = Vector3::new(-1f32, 2f32, -1f32);
        assert_eq!(actual, expected);
    }

    #[test]
    fn veirfies_vectors_linearly_independent() {
        let actual = lin_ind(
            &Vector3::UP(),
            &Vector3::RIGHT(),
            &Vector3::FOWARD(),
            0.001f32,
        );
        assert_eq!(actual, true);
    }

    #[test]
    fn veirfies_vectors_linearly_dependent() {
        let actual = lin_ind(
            &Vector3::UP(),
            &Vector3::ONE(),
            &Vector3::FOWARD(),
            0.001f32,
        );
        assert_eq!(actual, false);
    }

    #[test]
    fn verifies_cos_between_vecs() {
        let vec1 = Vector3::new(4f32, 3f32, 5f32);
        let vec2 = Vector3::new(3f32, 4f32, 5f32);
        assert_eq!(0.98f32, cos(&vec1, &vec2));
    }

    #[test]
    fn verifies_sin_between_vecs() {
        let vec1 = Vector3::new(4f32, 3f32, 5f32);
        let vec2 = Vector3::new(3f32, 4f32, 5f32);
        assert_eq!(0.19899738f32, sin(&vec1, &vec2));
    }

    #[test]
    fn dist_origin_point_one() {
        assert_eq!(1.7320508f32, dist(&Point3::origin(), &Point3::ONE()));
    }

    #[test]
    fn vector3_to_vec() {
        let vec = Vector3::new(4f32, 3f32, 5f32);
        let expected = vec![4f32, 3f32, 5f32];

        assert_eq!(expected, vec.to_vector());
    }

    #[test]
    fn transform_vector3_to_another_space_vector3() {
        let vec = Vector3::new(1f32, 2f32, 3f32);
        let transform_matrix = M::new_idx(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 5f32);
        let vec_transform_vec = Vector3::new(3f32, 4f32, 6f32);

        assert_eq!(
            Vector3::new(17f32, 36f32, 44f32),
            vec.transform(transform_matrix, vec_transform_vec)
        );
    }

    #[test]
    fn nonuniform_scale_by_1_2_3() {
        let vec = Vector3::ONE();
        let expected = Vector3::new(1f32, 2f32, 3f32);

        assert_eq!(expected, vec.nonuniform_scale(1f32, 2f32, 3f32));
    }

    #[test]
    fn indexing_0_returns_x() {
        let vector = Vector3::new(1f32, 2f32, 3f32);

        assert_eq!(vector[0usize], 1f32);
    }

    #[test]
    fn indexing_1_returns_y() {
        let vector = Vector3::new(1f32, 2f32, 3f32);

        assert_eq!(vector[1usize], 2f32);
    }

    #[test]
    fn indexing_2_returns_z() {
        let vector = Vector3::new(1f32, 2f32, 3f32);

        assert_eq!(vector[2usize], 3f32);
    }

    #[test]
    fn arithmetic() {
        let v = Vector3::ONE();

        assert_eq!(v.clone() / 2.0, Vector3::new(0.5f32, 0.5f32, 0.5f32));
        assert_eq!(v.clone() * 2.0, Vector3::new(2f32, 2f32, 2f32));
    }

    #[test]
    fn arithmetic_by_ref() {
        let v = Vector3::ONE();

        assert_eq!(&v / 2.0, Vector3::new(0.5f32, 0.5f32, 0.5f32));
        assert_eq!(&v * 2.0, Vector3::new(2f32, 2f32, 2f32));
    }

    #[test]
    fn dot_product_point() {
        let point = Point3 {
            x: 2f32,
            y: 1f32,
            z: 2f32,
        };
        let vec = Vector3 {
            x: 1f32,
            y: 2f32,
            z: 3f32,
        };
        let actual = &vec * &point;
        let expected = 10f32;
        assert_eq!(actual, expected);
    }
}
