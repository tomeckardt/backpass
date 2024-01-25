use std::{cell::RefCell, ops::{Add, Div, Mul, Sub}, rc::Rc};

use num_traits::One;

use crate::{Grad, GradVec};

pub struct Addition<T> {
    lhs_grad: Rc<dyn Grad<T>>, rhs_grad: Rc<dyn Grad<T>>
}

impl<T: One + Copy + Add<Output = T>> Grad<T> for Addition<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            self.lhs_grad.evaluate(dx)
                .into_iter()
                .zip(self.rhs_grad.evaluate(dx))
                .map(|(x, y)| x + y)
                .collect()
        }   
    }
}

impl<T: Copy + Add<Output = T> + One + 'static> Add<&GradVec<T>> for &GradVec<T> {

    type Output = GradVec<T>;

    fn add(self, rhs: &GradVec<T>) -> Self::Output {
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().copied().zip(rhs.data.borrow().iter().copied()).map(|(x, y)| x + y).collect()
            )),
            grad: Rc::new(Addition { lhs_grad: self.grad.clone(), rhs_grad: rhs.grad.clone() })
        }
    }
}

impl<T: Copy + Add<Output = T>> Add<T> for &GradVec<T> {

    type Output = GradVec<T>;

    fn add(self, rhs: T) -> Self::Output {
        let r = vec![rhs; self.data.borrow().len()];
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().copied().zip(r).map(|(x, y)| x + y).collect()
            )),
            grad: self.grad.clone()
        }
    }
}

pub struct Subtraction<T> {
    lhs_grad: Rc<dyn Grad<T>>, rhs_grad: Rc<dyn Grad<T>>
}

impl<T: One + Copy + Sub<Output = T>> Grad<T> for Subtraction<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            self.lhs_grad.evaluate(dx)
                .into_iter()
                .zip(self.rhs_grad.evaluate(dx))
                .map(|(x, y)| x - y)
                .collect()
        }   
    }
}

impl<T: Copy + Sub<Output = T> + One + 'static> Sub<&GradVec<T>> for &GradVec<T> {

    type Output = GradVec<T>;

    fn sub(self, rhs: &GradVec<T>) -> Self::Output {
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().copied().zip(rhs.data.borrow().iter().copied()).map(|(x, y)| x - y).collect()
            )),
            grad: Rc::new(Subtraction { lhs_grad: self.grad.clone(), rhs_grad: rhs.grad.clone() })
        }
    }
}

pub struct Multiplication<T> {
    lhs: Rc<RefCell<Vec<T>>>, rhs: Rc<RefCell<Vec<T>>>,
    lhs_grad: Rc<dyn Grad<T>>, rhs_grad: Rc<dyn Grad<T>>
}

impl<T: Add<Output = T> + Mul<Output = T> + One + Copy> Grad<T> for Multiplication<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            let dself: Vec<T> = self.lhs_grad.evaluate(dx).into_iter().zip(self.rhs.borrow().iter().copied()).map(|(x, y)| x * y).collect();
            let drhs: Vec<T> = self.lhs.borrow().iter().copied().zip(self.rhs_grad.evaluate(dx)).map(|(x, y)| x * y).collect();
            dself.into_iter().zip(drhs).map(|(x, y)| x + y).collect()
        }
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T> + One + 'static> Mul<&GradVec<T>> for &GradVec<T> {

    type Output = GradVec<T>;

    fn mul(self, rhs: &GradVec<T>) -> Self::Output {
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().copied().zip(rhs.data.borrow().iter().copied()).map(|(x, y)| x * y).collect()
            )),
            grad: Rc::new(Multiplication {
                lhs: self.data.clone(),
                rhs: rhs.data.clone(),
                lhs_grad: self.grad.clone(), 
                rhs_grad: rhs.grad.clone()
            })
        }
    }
}

pub struct Division<T> {
    lhs: Rc<RefCell<Vec<T>>>, rhs: Rc<RefCell<Vec<T>>>,
    lhs_grad: Rc<dyn Grad<T>>, rhs_grad: Rc<dyn Grad<T>>
}

impl<T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + One + Copy> Grad<T> for Division<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            let dself: Vec<T> = self.lhs_grad.evaluate(dx).into_iter().zip(self.rhs.borrow().iter().copied()).map(|(x, y)| x * y).collect();
            let drhs: Vec<T> = self.lhs.borrow().iter().copied().zip(self.rhs_grad.evaluate(dx)).map(|(x, y)| x * y).collect();
            dself.into_iter().zip(drhs).zip(self.rhs.borrow().iter().copied()).map(|((x, y), z)| (x - y) / z).collect()
        }
    }
}

impl<T: Copy + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + One + 'static> Div<&GradVec<T>> for &GradVec<T> {

    type Output = GradVec<T>;

    fn div(self, rhs: &GradVec<T>) -> Self::Output {
        GradVec {
            data: Rc::new(RefCell::new(
                self.data.borrow().iter().copied().zip(rhs.data.borrow().iter().copied()).map(|(x, y)| x * y).collect()
            )),
            grad: Rc::new(Division {
                lhs: self.data.clone(),
                rhs: rhs.data.clone(),
                lhs_grad: self.grad.clone(), 
                rhs_grad: rhs.grad.clone()
            })
        }
    }
}