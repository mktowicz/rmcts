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

use std::cell::RefCell;
use std::rc::Rc;
use std::rc::Weak;

use crate::state::State;

pub type NodeRef<T, S> = Rc<RefCell<Node<T, S>>>;

#[derive(Clone, Debug)]
pub struct Node<T, S>
where
    S: State<T>,
    T: Clone,
{
    pub action: T,
    pub state: S,
    pub visits: u32,
    pub total_reward: f32,
    pub expanded: bool,
    pub children: Vec<NodeRef<T, S>>,
    parent: Option<Weak<RefCell<Node<T, S>>>>,
}

impl<T, S> Node<T, S>
where
    S: State<T>,
    T: Clone,
{
    pub fn new(action: T, state: S) -> NodeRef<T, S> {
        Rc::new(RefCell::new(Self {
            action,
            state,
            visits: 0,
            total_reward: 0.,
            expanded: false,
            children: vec![],
            parent: None,
        }))
    }

    pub fn parent(&self) -> Option<NodeRef<T, S>> {
        if let Some(parent) = self.parent.clone() {
            parent.upgrade()
        } else {
            None
        }
    }

    pub fn set_parent(&mut self, node: &NodeRef<T, S>) {
        self.parent = Some(Rc::downgrade(node));
    }

    pub fn child_at(&self, index: usize) -> Option<NodeRef<T, S>> {
        if self.children.len() > index {
            Some(Rc::clone(&self.children[index]))
        } else {
            None
        }
    }

    pub fn best_child(&self) -> Option<NodeRef<T, S>> {
        match self.children.iter().max_by(|x, y| {
            x.borrow()
                .total_reward
                .partial_cmp(&y.borrow().total_reward)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Some(x) => Some(Rc::clone(x)),
            None => None,
        }
    }

    pub fn add_child(&mut self, node: NodeRef<T, S>) -> NodeRef<T, S> {
        self.children.push(node);
        Rc::clone(&self.children[self.children.len() - 1])
    }

    pub fn score(&self, c: f32) -> f32 {
        match self.parent() {
            Some(x) => {
                self.total_reward / self.visits as f32
                    + c * ((2. * (x.borrow().visits as f32).ln()) / self.visits as f32).sqrt()
            }
            None => 0.,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct DummyState {}

    impl State<u8> for DummyState {
        fn next_action(&self) -> Option<u8> {
            Some(1)
        }

        fn do_action(&mut self, _action: &u8) -> f32 {
            1.0
        }
    }

    impl DummyState {
        fn new() -> Self {
            Self {}
        }
    }

    fn build_1depth_tree(size: u8) -> NodeRef<u8, DummyState> {
        let mut state = DummyState::new();
        let action = state.next_action().unwrap();
        state.do_action(&action);

        let root = Node::new(action, state);

        for _i in 0..size {
            let mut state = root.borrow().state.clone();
            let action = state.next_action().unwrap();
            state.do_action(&action);

            let child = Node::new(action, state);
            child.borrow_mut().set_parent(&root);
            root.borrow_mut().add_child(child);
        }

        root
    }

    #[test]
    fn rc_counts() {
        let node = build_1depth_tree(5);
        assert_eq!(Rc::strong_count(&node), 1);
        assert_eq!(Rc::weak_count(&node), 5);
    }

    #[test]
    fn child_at() {
        let node = build_1depth_tree(5);
        assert!(node.borrow().child_at(2).is_some());

        let leaf = node.borrow().child_at(2).unwrap();
        assert!(leaf.borrow().child_at(2).is_none());
        assert!(leaf.borrow().parent().is_some());
    }

    #[test]
    fn best_child() {
        let node = build_1depth_tree(5);
        assert!(node.borrow().best_child().is_some());

        let leaf = node.borrow().child_at(2).unwrap();
        assert!(leaf.borrow().best_child().is_none());

        // Increase reward manually and check the node is selected
        leaf.borrow_mut().total_reward = 0.5;
        assert_eq!(
            node.borrow().best_child().unwrap().borrow().total_reward,
            0.5
        );
    }

    #[test]
    fn score() {
        let node = build_1depth_tree(5);
        assert_eq!(node.borrow().score(1.), 0.);

        // If the parent was not visited
        let leaf = node.borrow().child_at(2).unwrap();
        leaf.borrow_mut().visits = 1;
        leaf.borrow_mut().total_reward = 0.5;
        assert!(leaf.borrow().score(1.).is_nan());

        // If the parent has been visited
        node.borrow_mut().visits = 1;
        assert!(!leaf.borrow().score(1.).is_nan());
    }
}
