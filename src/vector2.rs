use std::ops;

///A 2D Vector with x and y coordinates: Vector2
#[derive(Clone, PartialEq, Debug)]
pub struct Vector2 {
    x: f32,
    y: f32,
}

///A 2D Point with x and y coordinates: Point2
#[derive(PartialEq, Debug)]
pub struct Point2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    ///Instantiates a new vector with to be defined values of x and y;
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 {x: x, y: y}
    }

    ///Instantiates a new Vector2 from 2 Point2 (initial position, final position).
    ///The new vector is created as final - initial (Points)
    pub fn diff(origin: Point2, destination: Point2) -> Vector2 {
        Vector2 {x: destination.x - origin.x, y: destination.y - origin.y}
    }

    ///Defines a Vector with UP direction (y=1, x=0)
    pub fn UP() -> Vector2 {
        Vector2 {x: 0f32, y: 1f32}
    }

    ///Defines a Vector with DOWN direction (y=-1, x=0)
    pub fn DOWN() -> Vector2 {
        Vector2 {x: 0f32, y: -1f32}
    }

    ///Defines a Vector with RIGHT direction (y=0, x=1)
    pub fn RIGHT() -> Vector2 {
        Vector2 {x: 1f32, y: 0f32}
    }

    ///Defines a Vector with LEFT direction (y=0, x=-1)
    pub fn LEFT() -> Vector2 {
        Vector2 {x: -1f32, y: 0f32}
    }

    ///Defines a 2D Vector with x=1 and y=1
    pub fn ONE() -> Vector2 {
        Vector2 {x: 1f32, y: 1f32}
    }

    ///Defines a Modulus ZERO Vector (x=0, y=0)
    pub fn ZERO() -> Vector2 {
        Vector2 {x: 0f32, y: 0f32}
    }

    ///Vector magnitude: the square root of the sum of each vector part to the power of 2
    pub fn magnitude(self) -> f32 {
        f32::sqrt(self.x.powi(2) + self.y.powi(2))
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

    ///Creates a new Vector2 relative to position (0, 0)
    pub fn to_vec(self) -> Vector2 {
        Vector2::diff(Point2::origin(), self)
    }

    ///Instantiates a Point2 with (0, 0)
    fn origin() -> Point2 {
        Point2::new(0f32, 0f32)
    }
}

impl ops::Add<Vector2> for Point2 {
    type Output = Point2;

    ///Overloads + for Points and Vectors: P + PQ = Q
    fn add(self, new_vec: Vector2) -> Point2 {
        Point2 {x: self.x + new_vec.x, y: self.y + new_vec.y}
    }
}

fn lin_ind(vec1: Vector2, vec2: Vector2) -> bool {
    vec1 * vec2 == 0f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vector2_with_parameters() {
        let actual = Vector2::new(1f32, 1f32);
        let expected = Vector2 {x: 1f32, y: 1f32};
        assert!(expected.x == actual.x &&
            expected.y == actual.y);
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
        assert!(expected.x == actual.x &&
            expected.y == actual.y);
    }

    #[test]
    fn creates_vector_from_point2() {
        let point = Point2::new(1f32, 1f32);
        let actual = point.to_vec();
        let expected = Vector2::ONE();
        assert!(expected.x == actual.x &&
            expected.y == actual.y);
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
        let actual = lin_ind(Vector2::UP(), Vector2::RIGHT());
        assert_eq!(actual, true);
    }

    #[test]
    fn veirfies_vectors_linearly_dependent() {
        let actual = lin_ind(Vector2::UP(), Vector2::ONE());
        assert_eq!(actual, false);
    }
}
