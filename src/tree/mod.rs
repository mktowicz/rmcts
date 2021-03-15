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

use std::rc::Rc;

use crate::node::{Node, NodeRef};
use crate::state::State;
use crate::strategies::{
    BackpropagationStrategy, ExpansionStrategy, SelectionStrategy, SimulationStrategy,
};

#[derive(Clone, Debug)]
pub struct Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    root: NodeRef<T, S>,
    learning_rate: f32,
    pub size: u32,
}

impl<T, S> Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    pub fn new(learning_rate: f32, action: T, state: S) -> Self {
        Self {
            root: Node::new(action, state),
            learning_rate,
            size: 1,
        }
    }

    pub fn root(&self) -> NodeRef<T, S> {
        Rc::clone(&self.root)
    }

    pub fn search(&mut self, iterations: u32) -> Option<NodeRef<T, S>> {
        for _i in 0..iterations {
            let mut leaf_node = match self.select() {
                Some(x) => x,
                None => break,
            };

            if leaf_node.borrow().visits > 0 {
                leaf_node = match self.expand(&mut leaf_node) {
                    Some(x) => x,
                    None => leaf_node,
                };
            }

            let reward = self.simulate(&leaf_node);
            self.backpropagate(&mut leaf_node, reward);
        }

        self.root.borrow().best_child()
    }

    pub fn add_node(&mut self, node: NodeRef<T, S>, parent: &mut NodeRef<T, S>) -> NodeRef<T, S> {
        self.size += 1;
        node.borrow_mut().set_parent(parent);
        parent.borrow_mut().add_child(node)
    }
}

impl<T, S> SelectionStrategy<T, S> for Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn select(&self) -> Option<NodeRef<T, S>> {
        let mut child = Rc::clone(&self.root);

        while child.borrow().children.len() > 0 {
            let next = match child.borrow().children.iter().max_by(|a, b| {
                if a.borrow().visits == 0 {
                    return std::cmp::Ordering::Greater;
                }

                a.borrow()
                    .score(self.learning_rate)
                    .partial_cmp(&b.borrow().score(self.learning_rate))
                    .unwrap_or(std::cmp::Ordering::Less)
            }) {
                Some(x) => Rc::clone(x),
                None => break,
            };

            child = next;
        }

        Some(child)
    }
}

impl<T, S> ExpansionStrategy<T, S> for Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn expand(&mut self, node: &mut NodeRef<T, S>) -> Option<NodeRef<T, S>> {
        let mut curr_state = node.borrow().state.clone();

        loop {
            if let Some(action) = curr_state.next_action() {
                let mut state = node.borrow().state.clone();
                state.do_action(&action);
                curr_state.do_action(&action);
                let new_node = Node::new(action, state);
                self.add_node(new_node, node);
            } else {
                break;
            }
        }

        node.borrow().child_at(0)
    }
}

impl<T, S> SimulationStrategy<T, S> for Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn simulate(&self, node: &NodeRef<T, S>) -> f32 {
        let mut total_reward = 0.0;
        let mut current_state = node.borrow().state.clone();

        loop {
            if let Some(action) = current_state.next_action() {
                total_reward += current_state.do_action(&action);
            } else {
                break;
            }
        }

        total_reward
    }
}

impl<T, S> BackpropagationStrategy<T, S> for Tree<T, S>
where
    S: State<T>,
    T: Clone,
{
    fn backpropagate(&mut self, node: &mut NodeRef<T, S>, value: f32) {
        let child = node;
        loop {
            child.borrow_mut().total_reward += value;
            child.borrow_mut().visits += 1;

            let parent = match child.borrow().parent() {
                Some(x) => x,
                None => break,
            };

            *child = parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct DummyState {
        action_reward: f32,
        actions: u8,
    }

    impl DummyState {
        fn new() -> Self {
            Self {
                action_reward: 0.5,
                actions: 5,
            }
        }
    }

    impl State<u8> for DummyState {
        fn next_action(&self) -> Option<u8> {
            if self.actions == 0 {
                return None;
            }
            Some(self.actions)
        }

        fn do_action(&mut self, _action: &u8) -> f32 {
            self.actions -= 1;
            self.action_reward
        }
    }

    #[test]
    fn expand() {
        let mut state1 = DummyState::new();
        let action1 = state1.next_action().unwrap();
        state1.do_action(&action1);
        let available_moves = state1.actions as usize;

        let mut tree = Tree::new(1.0, action1, state1);
        let node = tree.expand(&mut tree.root());

        assert!(node.is_some());
        assert!(node.unwrap().borrow().parent().is_some());
        assert!(tree.root().borrow().parent().is_none());
        assert_eq!(tree.root().borrow().children.len(), available_moves);
    }

    #[test]
    fn select() {
        let state1 = DummyState::new();
        let action1 = state1.next_action().unwrap();

        let mut tree = Tree::new(1.0, action1, state1);
        let node1 = tree.expand(&mut tree.root()).unwrap();

        node1.borrow_mut().visits = 1;
        node1.borrow_mut().total_reward = 1.;
        tree.root.borrow_mut().visits = 1;

        // Nodes that have not been visited before are favored
        let selected_node = tree.select().unwrap();
        assert_eq!(selected_node.borrow().total_reward, 0.0);
        assert_eq!(selected_node.borrow().visits, 0);
    }

    #[test]
    fn simulate() {
        let state1 = DummyState::new();
        let action1 = state1.next_action().unwrap();

        // Moves left * reward for each move
        let final_rerward = (state1.actions) as f32 * state1.action_reward;
        let tree = Tree::new(1.0, action1, state1);
        assert_eq!(tree.simulate(&mut tree.root()), final_rerward);
    }

    #[test]
    fn backpropagate() {
        let state1 = DummyState::new();
        let action1 = state1.next_action().unwrap();

        let mut tree = Tree::new(1.0, action1, state1);
        let mut node1 = tree.expand(&mut tree.root()).unwrap();
        let mut node2 = tree.expand(&mut node1).unwrap();

        tree.backpropagate(&mut node2, 5.0);
        assert_eq!(tree.root().borrow().total_reward, 5.0);
    }

    #[test]
    fn search() {
        let state1 = DummyState::new();
        let action1 = state1.next_action().unwrap();

        let mut tree = Tree::new(1.0, action1, state1);
        let best_node = tree.search(20).unwrap();

        for child in tree.root.borrow().children.iter() {
            assert!(child.borrow().total_reward <= best_node.borrow().total_reward);
        }
    }
}
