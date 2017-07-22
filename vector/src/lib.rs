#[derive(Debug)]
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
}
