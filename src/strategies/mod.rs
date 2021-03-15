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

use crate::node::NodeRef;
use crate::state::State;


pub trait SelectionStrategy<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn select(&self) -> Option<NodeRef<T, S>>;
}

pub trait ExpansionStrategy<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn expand(&mut self, node: &mut NodeRef<T, S>) -> Option<NodeRef<T, S>>;
}

pub trait RandomExpansionStrategy<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn expand(&mut self, node: &mut NodeRef<T, S>) -> Option<NodeRef<T, S>>;
}

pub trait SimulationStrategy<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn simulate(&self, node: &NodeRef<T, S>) -> f32;
}

pub trait BackpropagationStrategy<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn backpropagate(&mut self, node: &mut NodeRef<T, S>, value: f32);
}

