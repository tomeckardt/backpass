use std::ops::{Mul, Sub};

use crate::GradVec;


pub struct GradientDescent<'a, T> {
    params: Vec<&'a GradVec<T>>,
    lr: T
}

impl<'a, T: Copy + Mul<Output = T> + Sub<Output = T>> GradientDescent<'a, T> {
    pub fn new(lr: T, params: Vec<&'a GradVec<T>>) -> Self {
        Self { lr, params }
    }

    pub fn step(&self, v: &GradVec<T>) {
        for param in self.params.iter() {
            let grad = v.evaluate_grad(*param);
            let diff: Vec<_> = param.data.borrow().iter().zip(grad).map(|(&x, y)| x - self.lr * y).collect();
            *param.data.borrow_mut() = diff;
        }
    }
}

#[macro_export]
macro_rules! gd {
    (lr=$lr: expr, $($param: expr),*) => {
        GradientDescent::new($lr, vec![$($param),*])
    };
}