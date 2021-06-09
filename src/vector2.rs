use crate::math_helper;

use super::matrix::Matrix2x2 as M;

use std::ops;

///A 2D Vector with x and y coordinates: Vector2
#[derive(Clone, PartialEq, Debug)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

///A 2D Point with x and y coordinates: Point2
#[derive(PartialEq, Debug, Clone)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    ///Instantiates a new vector with to be defined values of x and y;
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 {x: x, y: y}
    }

    #[allow(dead_code)]
    ///Instantiates a new Vector2 from 2 Point2 (initial position, final position).
    ///The new vector is created as final - initial (Points)
    pub fn diff(origin: Point2, destination: Point2) -> Vector2 {
        Vector2 {x: destination.x - origin.x, y: destination.y - origin.y}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a Vector with UP direction (y=1, x=0)
    pub fn UP() -> Vector2 {
        Vector2 {x: 0f32, y: 1f32}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a Vector with DOWN direction (y=-1, x=0)
    pub fn DOWN() -> Vector2 {
        Vector2 {x: 0f32, y: -1f32}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a Vector with RIGHT direction (y=0, x=1)
    pub fn RIGHT() -> Vector2 {
        Vector2 {x: 1f32, y: 0f32}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a Vector with LEFT direction (y=0, x=-1)
    pub fn LEFT() -> Vector2 {
        Vector2 {x: -1f32, y: 0f32}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a 2D Vector with x=1 and y=1
    pub fn ONE() -> Vector2 {
        Vector2 {x: 1f32, y: 1f32}
    }

    #[allow(dead_code, non_snake_case)]
    ///Defines a Modulus ZERO Vector (x=0, y=0)
    pub fn ZERO() -> Vector2 {
        Vector2 {x: 0f32, y: 0f32}
    }

    #[allow(dead_code)]
    ///Vector magnitude: the square root of the sum of each vector part to the power of 2
    pub fn magnitude(self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2))
    }

    #[allow(dead_code)]
    ///Transforms a Vector 2 from one vectorspace to another via a matrix2x2 transform
    pub fn transform(self, m: M, vec: Vector2) -> Vector2 {
        (m * self) + vec
    }

    #[allow(dead_code)]
    ///Scales a Vector 2 in a non uniform way: (a, b * (x, y) = (ax, by)
    pub fn nonuniform_scale(self, a: f32, b: f32) -> Vector2 {
        let scale_matrix = M::new(Vector2::new(a, 0f32),
                                            Vector2::new(0f32, b));
        self.transform(scale_matrix, Vector2::ZERO())
    }
}

impl ops::Add for Vector2 {
    type Output = Vector2;

    ///Implements the Vector2 '+' trait
    fn add(self, new_vec: Vector2) -> Vector2 {
        Vector2 {x: self.x + new_vec.x, y: self.y + new_vec.y}
    }
}

impl ops::Mul<f32> for Vector2 {
    type Output = Vector2;

    ///Implements the scalar multiplication of a Vector2 with a f32. Other numbers should
    ///be passed with 'i as f32'
    fn mul(self, value: f32) -> Vector2 {
        Vector2 {x: self.x * value, y: self.y * value}
    }
}

impl ops::Mul<Vector2> for Vector2 {
    type Output = f32;

    ///Implements the dot product of 2 Vector2 as '*'.
    fn mul(self, new_vec: Vector2) -> f32 {
        self.x * new_vec.x + self.y * new_vec.y
    }
}

impl ops::Sub for Vector2 {
    type Output = Vector2;

    ///Implements the Vector2 '-' trait
    fn sub(self, new_vec: Vector2) -> Vector2 {
        Vector2 {x: self.x - new_vec.x, y: self.y - new_vec.y}
    }
}

impl Point2 {
    ///Instantiates a new Point2D with x and y.
    pub fn new(x: f32, y: f32) -> Point2 {
        Point2 {x: x, y: y}
    }

    #[allow(dead_code)]
    ///Creates a new Vector2 relative to position (0, 0)
    pub fn to_vec(self) -> Vector2 {
        Vector2::diff(Point2::origin(), self)
    }

    ///Instantiates a Point2 with (0, 0)
    fn origin() -> Point2 {
        Point2::new(0f32, 0f32)
    }

    #[allow(dead_code, non_snake_case)]
    ///Instantiates a Point2 with (1, 1)
    fn ONE() -> Point2 {
        Point2::new(1f32, 1f32)
    }
}

impl ops::Add<Vector2> for Point2 {
    type Output = Point2;

    ///Overloads + for Points and Vectors: P + PQ = Q
    fn add(self, new_vec: Vector2) -> Point2 {
        Point2 {x: self.x + new_vec.x, y: self.y + new_vec.y}
    }
}

#[allow(dead_code)]
///Vector2 linear indenpendency (2D)
fn lin_ind(vec1: Vector2, vec2: Vector2, delta: f32) -> bool {
    math_helper::float_eq(vec1 * vec2, 0f32, delta)
}

#[allow(dead_code)]
/// Cos between two vector2
fn cos(vec1: Vector2, vec2: Vector2) -> f32 {
    let dot_product = vec1.clone() * vec2.clone();
    let denominator = vec1.magnitude() * vec2.magnitude();
    dot_product / denominator
}

#[allow(dead_code)]
///Distance between 2 point2
fn dist(a: Point2, b: Point2) -> f32 {
    let x_dist = (a.x - b.x).powi(2);
    let y_dist = (a.y - b.y).powi(2);
    (x_dist + y_dist).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vector2_with_parameters() {
        let actual = Vector2::new(1f32, 1f32);
        let expected = Vector2 {x: 1f32, y: 1f32};
        assert_eq!(expected, actual);
    }

    #[test]
    fn creates_vector2_up() {
        let actual = Vector2::UP();
        assert!(actual.x == 0f32 &&
            actual.y == 1f32);
    }

    #[test]
    fn adds_right_and_left_vectors() {
         let actual = Vector2::RIGHT() + Vector2::LEFT();
         assert_eq!(actual.x, 0f32);
    }

    #[test]
    fn adds_right_and_up() {
        let actual = Vector2::RIGHT() + Vector2::UP();
        assert_eq!(actual.x, 1f32);
        assert_eq!(actual.y, 1f32);
    }

    #[test]
    fn mult_one_by_3() {
        let actual = Vector2::ONE() * 3f32;
        assert_eq!(actual.x, 3f32);
        assert_eq!(actual.y, 3f32);
    }

    #[test]
    fn sub_right_from_one() {
        let actual = Vector2::ONE() - Vector2::RIGHT();
        assert_eq!(actual.x, 0f32);
        assert_eq!(actual.y, 1f32);
    }

    #[test]
    fn magnitude_of_vector() {
        let vec = Vector2 {x: 3f32, y: 4f32};
        assert_eq!(vec.magnitude(), 5f32);
    }

    #[test]
    fn magnitude_of_vector_is_positive() {
        let vec = Vector2 {x: -3f32, y: 4f32};
        assert!(vec.magnitude() >= 0f32);
    }

    #[test]
    fn dot_product() {
        let vec1 = Vector2 {x: 2f32, y: 1f32};
        let vec2 = Vector2 {x: 1f32, y: 2f32};
        let actual = vec1 * vec2;
        let expected = 4f32;
        assert_eq!(actual, expected);
    }

    #[test]
    fn constructs_vector2_from_points2() {
        let vec = Vector2::diff(Point2 {x: 1f32, y: -1f32}, Point2 {x: 2f32, y: 3f32});
        assert_eq!(vec.x, 1f32);
        assert_eq!(vec.y, 4f32);
    }

    #[test]
    fn creates_point2_with_parameters() {
        let actual = Point2::new(1f32, 1f32);
        let expected = Point2 {x: 1f32, y: 1f32};
        assert_eq!(expected, actual);
    }

    #[test]
    fn creates_vector_from_point2() {
        let point = Point2::new(1f32, 1f32);
        let actual = point.to_vec();
        let expected = Vector2::ONE();
        assert_eq!(expected, actual);
    }

    #[test]
    fn point_add_vector_result_new_point() {
        let point = Point2::origin();
        let vec = Vector2::new(2f32, 3f32);
        let actual = point + vec;
        assert_eq!(actual.x, 2f32);
        assert_eq!(actual.y, 3f32);
    }

    #[test]
    fn veirfies_vectors_linearly_independent() {
        let actual = lin_ind(Vector2::UP(), Vector2::RIGHT(), 0.001f32);
        assert_eq!(actual, true);
    }

    #[test]
    fn veirfies_vectors_linearly_dependent() {
        let actual = lin_ind(Vector2::UP(), Vector2::ONE(), 0.001f32);
        assert_eq!(actual, false);
    }

    #[test]
    fn verifies_cos_between_vecs() {
        let vec1 = Vector2::new(4f32, 3f32);
        let vec2 = Vector2::new(3f32, 4f32);
        assert_eq!(0.96f32, cos(vec1, vec2));
    }

    #[test]
    fn dist_origin_point_one() {
        assert_eq!(1.4142135f32, dist(Point2::origin(), Point2::ONE()));
    }

    #[test]
    fn transform_vector3_to_another_space_vector3() {
        let vec = Vector2::new(1f32, 2f32);
        let transform_matrix = M::new_idx(1f32, 2f32, 3f32, 4f32);
        let vec_transform_vec = Vector2::new(3f32, 4f32);

        assert_eq!(Vector2::new(8f32, 15f32), vec.transform(transform_matrix, vec_transform_vec));
    }

    #[test]
    fn nonuniform_scale_by_4_3() {
        let vec = Vector2::ONE();
        let expected = Vector2::new(4f32, 3f32);

        assert_eq!(expected, vec.nonuniform_scale(4f32, 3f32));
    }
}
