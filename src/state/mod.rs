#![deny(
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

pub trait State<T>: Clone {
    fn next_action(&self) -> Option<T>;
    fn do_action(&mut self, action: &T) -> f32;
}


