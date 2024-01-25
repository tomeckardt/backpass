use std::{cell::RefCell, error::Error, fmt::Display, marker::PhantomData, rc::Rc};

use num_traits::{One, Zero};

pub trait Grad<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T>;
}

pub struct EmptyGrad<T> {
    data: PhantomData<T>
}

impl<T> EmptyGrad<T> {
    fn new() -> Self {
        Self { data: PhantomData }
    }
}

impl<T: Zero + One + Copy> Grad<T> for EmptyGrad<T> {
    fn evaluate(&self, dx: &GradVec<T>) -> Vec<T> {
        if self as *const dyn Grad<T> == Rc::as_ptr(&dx.grad) {
            vec![T::one(); dx.data.borrow().len()]
        } else {
            vec![T::zero(); dx.data.borrow().len()]
        }
    }
}

pub struct GradVec<T> {
    pub data: Rc<RefCell<Vec<T>>>,
    pub (super) grad: Rc<dyn Grad<T>>
}

impl<T: Zero + One + Copy + 'static, I: IntoIterator<Item = T>> From<I> for GradVec<T> {
    fn from(value: I) -> Self {
        Self {
            data: Rc::new(RefCell::new(value.into_iter().collect())),
            grad: Rc::new(EmptyGrad::new())
        }
    }
}

impl<T: Copy> GradVec<T> {
    pub fn evaluate_grad(&self, dx: &GradVec<T>) -> Vec<T> {
        self.grad.evaluate(dx)
    }
}

impl<T: Copy + One + Zero + 'static> GradVec<T> {
    pub fn from_scalar(gv: &GradVec<T>, value: T) -> GradVec<T> {
        GradVec { 
            data: Rc::new(RefCell::new(vec![value; gv.data.borrow().len()])), 
            grad: Rc::new(EmptyGrad::new()) 
        }
    }
}

#[derive(Debug)]
pub struct DiffLenErr {
    req: usize, act: usize
}
impl Display for DiffLenErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("given GradVec of length {}, but length {} is required", self.act, self.req))
    }
}
impl Error for DiffLenErr {}

impl<T: Zero + One + Copy + 'static> GradVec<T> {
    pub fn mutate(&mut self, data: Vec<T>) -> Result<(), DiffLenErr> {
        let self_len = self.data.borrow().len();
        let data_len = data.len();
        if self_len == data_len {
            self.data.replace(data);
            self.grad = Rc::new(EmptyGrad::new());
            Ok(())
        } else {
            Err(DiffLenErr { req: self_len, act: data_len })
        }
    }
}
