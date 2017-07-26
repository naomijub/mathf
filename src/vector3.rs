use std::ops;

///A 3D Vector with x, y, z coordinates: Vector3
#[derive(PartialEq, Debug)]
pub struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

///A 3D Point with x, y and z coordinates: Point3
#[derive(PartialEq, Debug)]
pub struct Point3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector3 {
    ///Instantiates a new vector with to be defined values of x, y, z;
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 {x: x, y: y, z: z}
    }

    ///Instantiates a new Vector3 from 2 Point3 (initial position, final position).
    ///The new vector is created as final - initial (Points)
    pub fn diff(origin: Point3, destination: Point3) -> Vector3 {
        Vector3 {x: destination.x - origin.x, y: destination.y - origin.y, z: destination.z - origin.z}
    }

    ///Defines a Vector with UP direction (y=1, x=0, z=0)
    pub fn UP() -> Vector3 {
        Vector3 {x: 0f32, y: 1f32, z: 0f32}
    }

    ///Defines a Vector with DOWN direction (y=-1, x=0, z=0)
    pub fn DOWN() -> Vector3 {
        Vector3 {x: 0f32, y: -1f32, z: 0f32}
    }

    ///Defines a Vector with RIGHT direction (y=0, x=1, z=0)
    pub fn RIGHT() -> Vector3 {
        Vector3 {x: 1f32, y: 0f32, z: 0f32}
    }

    ///Defines a Vector with LEFT direction (y=0, x=-1, z=0)
    pub fn LEFT() -> Vector3 {
        Vector3 {x: -1f32, y: 0f32, z: 0f32}
    }

    ///Defines a 3D Vector with x=1, y=1, z=1
    pub fn ONE() -> Vector3 {
        Vector3 {x: 1f32, y: 1f32, z: 1f32}
    }

    ///Defines a Modulus ZERO Vector (x=0, y=0, z=0)
    pub fn ZERO() -> Vector3 {
        Vector3 {x: 0f32, y: 0f32, z: 0f32}
    }

    ///Vector magnitude: the square root of the sum of each vector part to the power of 2
    pub fn magnitude(self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2) + self.z.powi(2))
    }
}

impl ops::Add for Vector3 {
    type Output = Vector3;

    ///Implements the Vector3 '+' trait
    fn add(self, new_vec: Vector3) -> Vector3 {
        Vector3 {x: self.x + new_vec.x, y: self.y + new_vec.y, z: self.z + new_vec.z}
    }
}

impl ops::Mul<f32> for Vector3 {
    type Output = Vector3;

    ///Implements the scalar multiplication of a Vector3 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn mul(self, value: f32) -> Vector3 {
        Vector3 {x: self.x * value, y: self.y * value, z: self.z * value}
    }
}

impl ops::Mul<Vector3> for Vector3 {
    type Output = f32;

    ///Implements the dot product of 2 Vector3 as '*'.
    fn mul(self, new_vec: Vector3) -> f32 {
        self.x * new_vec.x + self.y * new_vec.y + self.z * new_vec.z
    }
}

impl ops::Sub for Vector3 {
    type Output = Vector3;

    ///Implements the Vector3 '-' trait
    fn sub(self, new_vec: Vector3) -> Vector3 {
        Vector3 {x: self.x - new_vec.x, y: self.y - new_vec.y, z: self.z - new_vec.z}
    }
}

impl Point3 {
    ///Instantiates a new Point3 with x, y and z.
    pub fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3 {x: x, y: y, z: z}
    }

    ///Creates a new Vector3 relative to position (0, 0, 0)
    pub fn to_vec(self) -> Vector3 {
        Vector3::diff(Point3::origin(), self)
    }

    ///Instantiates a Point3 with (0, 0, 0)
    fn origin() -> Point3 {
        Point3::new(0f32, 0f32, 0f32)
    }
}

impl ops::Add<Vector3> for Point3 {
    type Output = Point3;

    ///Overloads + for Points and Vectors: P + PQ = Q
    fn add(self, new_vec: Vector3) -> Point3 {
        Point3 {x: self.x + new_vec.x, y: self.y + new_vec.y, z: self.z + new_vec.z}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vector3_with_parameters() {
        let actual = Vector3::new(1f32, 1f32, 1f32);
        let expected = Vector3 {x: 1f32, y: 1f32, z: 1f32};
        assert!(expected.x == actual.x &&
            expected.y == actual.y &&
            expected.z == actual.z);
    }

    #[test]
    fn creates_vector3_up() {
        let actual = Vector3::UP();
        assert!(actual.x == 0f32 &&
            actual.y == 1f32 &&
            actual.z == 0f32);
    }

    #[test]
    fn adds_right_and_left_vectors() {
         let actual = Vector3::RIGHT() + Vector3::LEFT();
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
    fn sub_right_from_one() {
        let actual = Vector3::ONE() - Vector3::RIGHT();
        assert_eq!(actual.x, 0f32);
        assert_eq!(actual.y, 1f32);
        assert_eq!(actual.z, 1f32);
    }

    #[test]
    fn magnitude_of_vector() {
        let vec = Vector3 {x: 1f32, y: 2f32, z: 3f32};
        assert_eq!(vec.magnitude(), 3.7416575f32);
    }

    #[test]
    fn magnitude_of_vector_is_positive() {
        let vec = Vector3 {x: -3f32, y: 4f32, z: 5f32};
        assert!(vec.magnitude() >= 0f32);
    }

    #[test]
    fn dot_product() {
        let vec1 = Vector3 {x: 2f32, y: 1f32, z: 2f32};
        let vec2 = Vector3 {x: 1f32, y: 2f32, z: 3f32};
        let actual = vec1 * vec2;
        let expected = 10f32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn constructs_vector3_from_points3() {
        let vec = Vector3::diff(Point3::new(1f32, -1f32, 2f32),
            Point3::new(2f32, 3f32, 2f32));
        assert_eq!(vec.x, 1f32);
        assert_eq!(vec.y, 4f32);
        assert_eq!(vec.z, 0f32);
    }

    #[test]
    fn creates_vector_from_point3() {
        let point = Point3::new(1f32, 1f32, 1f32);
        let actual = point.to_vec();
        let expected = Vector3::ONE();
        assert!(expected.x == actual.x &&
            expected.y == actual.y &&
            expected.z == actual.z);
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
}
