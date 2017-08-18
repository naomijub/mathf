use super::vector3::Vector3;
use std::ops;

///Implements a matrix 3 x 3
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3 {
    pub r1: Vector3,
    pub r2: Vector3,
    pub r3: Vector3,
}

impl Matrix3x3 {

    ///Creates a new Matrix3x3 from 3 vector3 rows
    pub fn new(r1: Vector3, r2: Vector3, r3: Vector3) -> Matrix3x3 {
        Matrix3x3 {r1: r1, r2: r2, r3: r3}
    }

    ///Creates a new Matrix3x3 from 9 indexed floats
    pub fn new_idx(n1: f32, n2: f32, n3: f32, n4: f32, n5: f32, n6: f32, n7: f32, n8: f32, n9: f32)
                                                                                    -> Matrix3x3 {

        Matrix3x3 {r1: Vector3::new(n1, n2, n3),
                   r2: Vector3::new(n4, n5, n6),
                   r3: Vector3::new(n7, n8, n9)}
    }

    pub fn det(self) -> f32 {
        (self.r1.x * self.r2.y * self.r3.z +
            self.r1.y * self.r2.z * self.r3.x +
            self.r1.z * self.r2.x * self.r3.y)
        - (self.r1.z * self.r2.y * self.r3.x +
            self.r1.y * self.r2.x * self.r3.z +
            self.r1.x * self.r2.z * self.r3.y)
    }
}

impl ops::Mul<Vector3> for Matrix3x3 {
    type Output = Vector3;

    ///Implements the transform matrix of a vector 3 into another vector 3.
    fn mul(self, vec: Vector3) -> Vector3 {
        Vector3 {x: self.r1 * vec.clone(),
            y: self.r2 * vec.clone(),
            z: self.r3 * vec.clone()}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_created_new_from_3_vector3() {
        let actual = Matrix3x3::new(Vector3::new(1f32, 2f32, 3f32),
                                        Vector3::new(1f32, 2f32, 3f32),
                                        Vector3::new(1f32, 2f32, 3f32));
        let expected = Matrix3x3 {r1: Vector3::new(1f32, 2f32, 3f32),
                                        r2: Vector3::new(1f32, 2f32, 3f32),
                                        r3: Vector3::new(1f32, 2f32, 3f32)};
        assert_eq!(expected, actual);
    }

    #[test]
    fn matrix_created_new_from_9_floats() {
        let actual = Matrix3x3::new_idx(1f32, 1f32, 1f32, 1f32, 1f32, 1f32,
                                    1f32, 1f32, 1f32);
        let expected = Matrix3x3 {r1: Vector3::new(1f32, 1f32, 1f32),
                                            r2: Vector3::new(1f32, 1f32, 1f32),
                                            r3: Vector3::new(1f32, 1f32, 1f32)};
        assert_eq!(expected, actual);
    }

    #[test]
    fn vector_transform_by_matrix() {
        let vec = Vector3::new(1f32, 2f32, 3f32);
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32,
                                        4f32, 5f32, 6f32, 7f32, 8f32, 9f32);
        let actual = matrix * vec;
        let expected = Vector3::new(14f32, 32f32, 50f32);
        assert_eq!(expected, actual);
    }

    #[test]
    fn seq_maxtrix_det() {
        let matrix = Matrix3x3::new_idx(1f32, 2f32, 3f32,
                                        4f32, 5f32, 6f32, 7f32, 8f32, 5f32);
        let actual = matrix.det();
        let expected = 12f32;
        assert_eq!(expected, actual);
    }
}