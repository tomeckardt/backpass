use std::{cell::RefCell, ops::Neg, rc::Rc};

use num_traits::{One, Signed, Zero};

use crate::{Grad, GradVec};

pub struct Negation<T> {
    grad: Rc<dyn Grad<T>>
}

impl<T: Copy + One + Neg<Output = T>> Grad<T> for Negation<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) { vec![T::one(); dx.data.borrow().len()] }
        else { self.grad.evaluate(dx).into_iter().map(T::neg).collect() }
    }
}

impl<T: Copy + One + Neg<Output = T> + 'static> Neg for &GradVec<T> {

    type Output = GradVec<T>;

    fn neg(self) -> Self::Output {
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().map(|&x| -x).collect()
            )),
            grad: Rc::new(Negation { grad: self.grad.clone() })
        }
    }
} 

pub struct Absolute<T> {
    grad: Rc<dyn Grad<T>>
}

impl<T: Copy + PartialOrd + One + Zero + Neg<Output = T>> Grad<T> for Absolute<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            let zero = T::zero();
            let grad = self.grad.evaluate(dx);
            dx.data.as_ref().borrow()
                .iter()
                .copied()
                .zip(grad)
                .map(|(x, y)| {
                    match x.partial_cmp(&zero) {
                        Some(std::cmp::Ordering::Less) => -y,
                        _ => y
                    }
                }).collect()
        }
    }
}

impl<T: Signed + Copy + PartialOrd + 'static> GradVec<T> {
    pub fn abs(&self) -> GradVec<T> {
        let data = Rc::new(RefCell::new(
            self.data.borrow().iter().map(T::abs).collect()
        ));
        let grad = Rc::new(
            Absolute { grad: self.grad.clone() }
        );
        GradVec { data, grad }
    }
}

macro_rules! impl_func {
    ($s: ident, $f: ident, $deriv: expr; $($t: ty),*) => {
        pub struct $s<T> {
            grad: Rc<dyn Grad<T>>,
            prev_value: Rc<RefCell<Vec<T>>>
        }
        $(
            impl Grad<$t> for $s<$t> {
                fn evaluate(&self, dx: &GradVec<$t>) -> Vec<$t> {
                    self.grad.evaluate(dx).into_iter().zip(self.prev_value.borrow().iter().copied()).map($deriv).collect()
                }
            }
            impl GradVec<$t> {
                pub fn $f(&self) -> Self {
                    let data = Rc::new(RefCell::new(
                        self.data.borrow().iter().copied().map(<$t>::$f).collect()
                    ));
                    let grad = Rc::new($s { 
                        grad: self.grad.clone(),
                        prev_value: self.data.clone()
                    });
                    Self { data, grad }
                }
            }
        )*
    };
}
impl_func!(Exp, exp, |(x, y)| x * y.exp(); f32, f64);
impl_func!(Sin, sin, |(x, y)| x * y.cos(); f32, f64);
impl_func!(Cos, cos, |(x, y)| -x * y.sin(); f32, f64);
impl_func!(Ln, ln, |(x, y)| x / y; f32, f64);