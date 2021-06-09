pub fn float_eq(a: f32, b: f32, delta: f32) -> bool {
    a == b || (a.abs() - b.abs()).abs().le(&delta)
}
