use std::ops;

#[derive(PartialEq, Debug)]
pub struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 {x: x, y: y}
    }

    pub fn UP() -> Vector2 {
        Vector2 {x: 0f32, y: 1f32}
    }

    pub fn DOWN() -> Vector2 {
        Vector2 {x: 0f32, y: -1f32}
    }

    pub fn RIGHT() -> Vector2 {
        Vector2 {x: 1f32, y: 0f32}
    }

    pub fn LEFT() -> Vector2 {
        Vector2 {x: -1f32, y: 0f32}
    }

    pub fn ONE() -> Vector2 {
        Vector2 {x: 1f32, y: 1f32}
    }

    pub fn ZERO() -> Vector2 {
        Vector2 {x: 0f32, y: 0f32}
    }
}

impl ops::Add for Vector2 {
    type Output = Vector2;

    fn add(self, new_vec: Vector2) -> Vector2 {
        Vector2 {x: self.x + new_vec.x, y: self.y + new_vec.y}
    }
}

impl ops::Mul<f32> for Vector2 {
    type Output = Vector2;

    fn mul(self, value: f32) -> Vector2 {
        Vector2 {x: self.x * value, y: self.y * value}
    }
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
    fn creates_vector2UP() {
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
}
